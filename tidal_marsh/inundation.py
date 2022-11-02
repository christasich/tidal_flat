from dataclasses import dataclass, field, InitVar
import numpy as np

import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from sklearn.utils import Bunch


@dataclass
class Inundation:
    depth: PchipInterpolator
    start: pd.Timestamp
    end: pd.Timestamp
    flood: pd.Series = field(init=False)
    ebb: pd.Series = field(init=False)
    slack: pd.Series = field(init=False)
    hydroperiod: pd.Timedelta = field(init=False)
    solve_ivp_kwargs: dict = field(init=False, default_factory=dict)
    result: pd.DataFrame = field(init=False, default=None)
    aggradation: float = 0.0
    valid: bool = False

    def __post_init__(self):
        self.logger = logger.bind(model_time=self.start)
        self.hydroperiod = self.end - self.start

        d1dt_roots = self.depth.derivative().roots()
        i = self.depth(d1dt_roots).argmax()
        t = pd.to_datetime(d1dt_roots[i], unit="s", origin="unix", utc=True).tz_convert(self.start.tz)
        self.slack = pd.Series(
            data=[t, self.depth(d1dt_roots[i])],
            index=["datetime", "depth"],
        )

        self.flood = pd.Series(
            data=[self.start, self.slack.datetime, self.slack.datetime - self.start],
            index=["start", "end", "period"],
            name="flood",
        )
        self.ebb = pd.Series(
            data=[self.slack.datetime, self.end, self.end - self.slack.datetime],
            index=["start", "end", "period"],
            name="ebb",
        )
        self.solve_ivp_kwargs = {
            "method": "RK45",
            "dense_output": False,
            "first_step": None,
            "max_step": np.inf,
            "rtol": 1e-3,
            "atol": 1e-6,
        }

    @staticmethod
    def solve_flood(t, y, depth, ssc_boundary, bulk_density, settling_rate):
        concentration = y[0]

        d1dt_aggradation = settling_rate * concentration / bulk_density
        d1dt_concentration = -(settling_rate * concentration) / depth(t) - 1 / depth(t) * (
            concentration - ssc_boundary
        ) * depth.derivative()(t)

        return [d1dt_concentration, d1dt_aggradation]

    @staticmethod
    def solve_ebb(t, y, depth, bulk_density, settling_rate):
        concentration = y[0]

        d1dt_aggradation = settling_rate * concentration / bulk_density
        d1dt_concentration = -(settling_rate * concentration) / depth(t)

        return [d1dt_concentration, d1dt_aggradation]

    def integrate(self, ssc_boundary, bulk_density, settling_rate, solve_ivp_kwargs=None):
        if solve_ivp_kwargs is not None:
            self.solve_ivp_kwargs |= solve_ivp_kwargs

        self.logger.trace("Integrating flood limb.")
        params = Bunch(
            depth=self.depth, ssc_boundary=ssc_boundary, bulk_density=bulk_density, settling_rate=settling_rate
        )
        flood = solve_ivp(
            fun=self.solve_flood,
            t_span=[self.flood.start.timestamp(), self.flood.end.timestamp()],
            y0=(0.0, 0.0),
            args=[params.depth, params.ssc_boundary, params.bulk_density, params.settling_rate],
            **self.solve_ivp_kwargs,
        )
        flood.params = params
        self.validate(flood)

        self.logger.trace("Integrating ebb limb.")
        ebb = solve_ivp(
            fun=self.solve_ebb,
            t_span=[self.ebb.start.timestamp(), self.ebb.end.timestamp()],
            y0=(flood.y[0][-1], flood.y[1][-1]),
            args=[params.depth, params.bulk_density, params.settling_rate],
            **self.solve_ivp_kwargs,
        )
        ebb.params = params
        self.validate(ebb)
        if flood.valid and ebb.valid:
            self.valid = True

        self.concat_results(flood, ebb)
        self.aggradation = self.result.aggradation.iat[-1]

    def validate(self, result) -> None:
        result.valid = True
        if result.success is False:
            result.valid = False
            self.logger.warning(result.message)
        if (result.y[1] < 0.0).any():
            result.valid = False
            self.logger.warning("Solution contains negative aggradations.")
        if result.y[1][-1] > self.slack.depth * result.params.ssc_boundary:
            result.valid = False
            self.logger.warning("More sediment extracted than possible.")

    def concat_results(self, flood, ebb):
        time = np.append(flood.t, ebb.t[1:])
        concentration = np.append(flood.y[0], ebb.y[0, 1:])
        aggradation = np.append(flood.y[1], ebb.y[1, 1:])
        depth = self.depth(time)
        index = pd.to_datetime(time, unit="s", origin="unix", utc=True).tz_convert(self.start.tz)
        self.result = pd.DataFrame(
            data={
                "depth": depth,
                "concentration": concentration,
                "aggradation": aggradation,
            },
            index=index,
        )

    def summarize(self):
        return pd.Series(
            data={
                "start": self.start,
                "end": self.end,
                "hydroperiod": self.hydroperiod,
                "aggradation": self.aggradation,
                "depth": self.slack.depth,
                "valid": self.valid,
            }
        )

    def plot(self):

        mosaic = [["e", "i"], ["c", "i"], ["d", "i"]]

        fig, ax = plt.subplot_mosaic(mosaic=mosaic, figsize=(12, 8), sharex=True, gridspec_kw={"width_ratios": [5, 2]})

        index = pd.date_range(self.flood.start, self.ebb.end, freq="10S")
        values = self.depth(index.astype(int) / 10**9)
        # values = self.params.depth_spl((index - self.start).total_seconds().values) + self.initial_elevation
        # water_levels = pd.Series(data=values, index=index)
        sns.lineplot(x=index, y=values, color="cornflowerblue", label="Tide", ax=ax["e"])
        # sns.lineplot(data=self.tides.loc[self.start:self.end] - self.initial_elevation, color="red", ax=ax["e"])
        sns.lineplot(data=self.result.aggradation, color="black", ls=":", label="Land Surface", ax=ax["e"])
        ax["e"].set_ylabel(ylabel="Depth (m)")

        sns.lineplot(data=self.result.concentration, color="saddlebrown", label="SSC", ax=ax["c"])
        ax["c"].set_ylabel(ylabel="Concentration (g/L)")

        sns.lineplot(data=self.result.aggradation * 1000, label="Aggradation", color="green", ax=ax["d"])
        ax["d"].set_ylabel(ylabel="Height (mm)")
        # ax["d"].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        data = {
            "start": self.start.round("S"),
            "end": self.end.round("S"),
            "period": (
                f"{self.hydroperiod.components.hours:02}H {self.hydroperiod.components.minutes:02}M"
                f" {self.hydroperiod.components.seconds:02}S"
            ),
            # "aggradation": f"{self.aggradation:.2e}",
            # "degradation": f"{self.subsidence:.2e}",
            # "$\Delta$elevation": f"{self.aggradation + self.subsidence:.2e}",
            **self.solve_ivp_kwargs,
        }
        info = pd.DataFrame(data=data.values(), index=data.keys())

        for v in ["e", "c", "d"]:
            a = ax[v]
            a.axvline(self.slack.datetime, color="black", linestyle="--")
            a.ticklabel_format(axis="y", useOffset=False)

        ax["i"].table(cellText=info.values, rowLabels=info.index, cellLoc="center", bbox=[0.25, 0.25, 0.5, 0.5])
        ax["i"].axis("off")

        ax["e"].text(x=self.slack.datetime, y=ax["e"].get_ylim()[1], s="slack", ha="center")
        plt.xticks(rotation=45)
