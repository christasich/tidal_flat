from dataclasses import InitVar, dataclass, field

import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.utils import Bunch

from . import utils


class InundationResult(OdeResult):
    pass


@dataclass
class Inundation:
    water_levels: pd.Series
    initial_elevation: float
    ssc_boundary: InitVar[float]
    bulk_density: InitVar[float]
    settling_rate: InitVar[float]
    constant_rates: InitVar[float]
    result: None | InundationResult = field(default=None, init=False)
    data: pd.DataFrame = field(default=None, init=False)
    aggradation_total: float = field(default=None, init=False)
    degradation_total: float = field(default=None, init=False)

    def __post_init__(self, ssc_boundary, bulk_density, settling_rate, constant_rates):
        self.start = self.water_levels.index[0]
        self.end = self.water_levels.index[-1]
        self.timestep = self.water_levels.index[1] - self.water_levels.index[0]
        self.period = self.end - self.start
        self.slack = self.water_levels.idxmax()
        self.slack_elevation = self.water_levels.max()
        tides_func = InterpolatedUnivariateSpline(
            x=utils.datetime2num(self.water_levels.index), y=self.water_levels.values, k=3
        )
        self.params = Bunch(
            tides_func=tides_func,
            ssc_boundary=ssc_boundary,
            bulk_density=bulk_density,
            settling_rate=settling_rate,
            constant_rates=constant_rates,
        )
        self.overextraction = 0.0

    def calculate_elevation(self, at=None, to=None):
        if at:
            elapsed_seconds = (at - self.now).total_seconds()
            return self.constant_rates * elapsed_seconds + self.elevation
        elif to:
            index = self.water_levels.loc[self.now : to].index
            elapsed_seconds = (index - self.now).total_seconds()
            return (self.constant_rates * elapsed_seconds).values + self.elevation

    @staticmethod
    def solve_odes(t, y, params):
        water_level = y[0]
        concentration = y[1]
        elevation = y[2]
        depth = water_level - elevation

        d1dt_water_level = params.tides_func.derivative()(t)
        d1dt_aggradation = params.settling_rate * concentration / params.bulk_density
        d1dt_degradation = params.constant_rates
        d1dt_elevation = d1dt_aggradation + d1dt_degradation
        d1dt_depth = d1dt_water_level - d1dt_elevation
        if d1dt_depth > 0:
            d1dt_concentration = (
                -(params.settling_rate * concentration) / depth
                - 1 / depth * (concentration - params.ssc_boundary) * d1dt_depth
            )
            d1dt_aggradation_max = params.ssc_boundary * d1dt_depth / params.bulk_density
        else:
            d1dt_concentration = -(params.settling_rate * concentration) / depth
            d1dt_aggradation_max = 0.0

        return [
            d1dt_water_level,
            d1dt_concentration,
            d1dt_elevation,
            d1dt_aggradation,
            d1dt_aggradation_max,
            d1dt_degradation,
        ]  # 0  # 1  # 2  # 3  # 4

    def zero_conc(t, y, params):
        return y[1]  # - 1e-5

    zero_conc.terminal = True
    zero_conc.direction = -1

    zero_conc = staticmethod(zero_conc)

    def zero_depth(t, y, params):
        return y[0] - y[2]  # - 1e-5

    zero_depth.terminal = True
    zero_depth.direction = -1

    zero_depth = staticmethod(zero_depth)

    def integrate(self, method="DOP853", dense_output=False):

        self.result = solve_ivp(
            fun=self.solve_odes,
            t_span=[utils.datetime2num(self.start), utils.datetime2num(self.end)],
            y0=[self.water_levels.values[0], 0.0, self.initial_elevation, 0.0, 0.0, 0.0],
            method=method,
            events=(self.zero_conc, self.zero_depth),
            dense_output=dense_output,
            atol=(1e-6, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8),
            args=[self.params],
        )

        self.aggradation_total = self.result.y[3][-1]
        self._set_df()
        self._validate_result()

    def _validate_result(self):
        if self.result.success is False:
            logger.warning(f"{self.start} | Integration failed!\nparams={self.params}")
        if (self.data.aggradation < 0.0).any():
            logger.warning(f"{self.start} | Negative aggradation detected!\n{self.data.loc[self.data.aggradation < 0]}")
        if (self.data.aggradation_max < self.data.aggradation).any():
            logger.warning(
                f"{self.start} | Overextraction detected. Extracted {self.data.aggradation.values[-1] / self.data.aggradation_max.values[-1] * 100:.2f}% of possible."
            )
            self.overextraction = self.data.aggradation.values[-1] - self.data.aggradation_max.values[-1]

    def _set_df(self):
        index = utils.num2datetime(self.result.t)
        water_level = self.result.y[0]
        concentration = self.result.y[1]
        elevation = self.result.y[2]
        aggradation = self.result.y[3]
        aggradation_max = self.result.y[4]
        degradation = self.result.y[5]
        df = pd.DataFrame(
            data={
                "water_level": water_level,
                "elevation": elevation,
                "concentration": concentration,
                "aggradation": aggradation,
                "aggradation_max": aggradation_max,
                "degradation": degradation,
            },
            index=index,
        )
        if index[-1] != self.end:
            record = {}
            record["water_level"] = self.water_levels.at[self.end]
            remaining_seconds = (self.end - df.index[-1]).total_seconds()
            record["elevation"] = df.elevation.values[-1] + remaining_seconds * self.params.constant_rates
            record["concentration"] = 0.0
            record["aggradation"] = df.aggradation.values[-1]
            record["aggradation_max"] = df.aggradation_max.values[-1]
            record["degradation"] = df.degradation.values[-1]
            df.loc[self.end] = record

        self.degradation_total = df.degradation.values[-1]
        self.data = df

    def plot(self):

        fig, ax = plt.subplots(figsize=(15, 15), nrows=4, ncols=1, constrained_layout=True)
        fig.suptitle(f"Inundation at {self.start}", fontsize=16)

        aggr_max_mod_diff = self.data.aggradation_max - self.data.aggradation

        sns.lineplot(data=self.data.water_level, color="cornflowerblue", label="Tide", ax=ax[0])
        sns.lineplot(data=self.data.elevation, color="forestgreen", label="Land Surface", ax=ax[0])
        ax[0].set_ylabel(ylabel="Elevation (m)")

        sns.lineplot(data=self.data.concentration, color="saddlebrown", label="SSC", ax=ax[1])
        ax[1].set_ylabel(ylabel="Concentration (g/L)")

        sns.lineplot(data=self.data.aggradation_max, color="red", label="Max", ax=ax[2])
        sns.scatterplot(data=self.data.aggradation, label="Modeled", ax=ax[2])
        ax[2].set_ylabel(ylabel="Aggradation (m)")
        ax[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        sns.lineplot(data=aggr_max_mod_diff, color="black", linestyle=":", ax=ax[3])
        ax[3].set_ylabel(ylabel="Difference (m)\nMax - Modeled")
        ax[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax[3].fill_between(
            x=self.data.index, y1=aggr_max_mod_diff, where=aggr_max_mod_diff >= 0.0, color="forestgreen", alpha=0.3
        )
        ax[3].fill_between(x=self.data.index, y1=aggr_max_mod_diff, where=aggr_max_mod_diff < 0, color="red", alpha=0.3)
        for ax in ax:
            ax.axvline(self.slack, color="black", linestyle="--")
            ax.ticklabel_format(axis="y", useOffset=False)
