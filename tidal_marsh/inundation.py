from dataclasses import InitVar, dataclass, field
from textwrap import wrap
import numpy as np

import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.utils import Bunch
from . import utils

@dataclass
class Inundation:
    water_levels: pd.Series
    initial_elevation: float
    ssc_boundary: float
    bulk_density: float
    settling_rate: float
    constant_rates: float
    solve_ivp_opts: dict
    depth_spl: InterpolatedUnivariateSpline = field(init=False)
    flood: OdeResult = field(init=False, default=None)
    ebb: OdeResult = field(init=False, default=None)
    valid: bool = True
    result: pd.DataFrame = field(init=False, default=None)
    total_aggradation: float = 0.0
    total_subsidence: float = 0.0

    def __post_init__(self):
        self.start = self.water_levels.index[0]
        self.end = self.water_levels.index[-1]
        self.logger = logger.bind(model_time=self.start)
        self.timestep = self.water_levels.index[1] - self.water_levels.index[0]
        self.period = self.end - self.start
        self.slack = self.water_levels.idxmax()
        self.slack_depth = self.water_levels.max() - self.initial_elevation
        time = (self.water_levels.index-self.water_levels.index[0]).total_seconds().values
        depth = self.water_levels.values - self.initial_elevation
        self.depth_spl = InterpolatedUnivariateSpline(x=time, y=depth, k=3)

        self.params = Bunch(
            depth_spl=self.depth_spl,
            ssc_boundary=self.ssc_boundary,
            bulk_density=self.bulk_density,
            settling_rate=self.settling_rate,
            constant_rates=self.constant_rates,
        )

    @staticmethod
    def solve_flood(t, y, params):
        concentration = y[0]

        d1dt_aggradation = params.settling_rate * concentration / params.bulk_density
        d1dt_concentration = (
            -(params.settling_rate * concentration) / params.depth_spl(t) - 1 /
            params.depth_spl(t) * (concentration - params.ssc_boundary) * params.depth_spl.derivative()(t)
            )

        return [d1dt_concentration, d1dt_aggradation]

    @staticmethod
    def solve_ebb(t, y, params):
        concentration = y[0]

        d1dt_aggradation = params.settling_rate * concentration / params.bulk_density
        d1dt_concentration = -(params.settling_rate * concentration) / params.depth_spl(t)

        return [d1dt_concentration, d1dt_aggradation]

    def integrate(self):
        self.logger.trace('Integrating flood limb.')
        flood_t_span = (0.0, (self.slack - self.start).total_seconds())
        self.flood = solve_ivp(
            fun=self.solve_flood,
            t_span=flood_t_span,
            y0=(0.0, 0.0),
            args=[self.params],
            **self.solve_ivp_opts,
        )
        self.logger.trace('Integrating ebb limb.')
        ebb_t_span = ((self.slack - self.start).total_seconds(), (self.end - self.start).total_seconds())
        self.ebb = solve_ivp(
                fun=self.solve_ebb,
                t_span=ebb_t_span,
                y0=(self.flood.y[0][-1], self.flood.y[1][-1]),
                args=[self.params],
                **self.solve_ivp_opts,
            )
        self.validate()
        self.concat_results()

    def validate(self) -> None:
        self.valid = True
        for limb in ['flood', 'ebb']:
            result = eval('self.' + limb)
            result.valid = True
            if result.success is False:
                self.valid = result.valid = False
                self.logger.warning(result.message)
            if (result.y[1] < 0.0).any():
                self.valid = result.valid = False
                self.logger.warning("Solution contains negative aggradations.")
            if result.y[1][-1] > self.slack_depth * self.params.ssc_boundary:
                self.valid = result.valid = False
                self.logger.warning("More sediment extracted than possible.")

    def concat_results(self):
        time = np.append(self.flood.t, self.ebb.t[1:])
        concentration = np.append(self.flood.y[0], self.ebb.y[0, 1:])
        aggradation = np.append(self.flood.y[1], self.ebb.y[1, 1:])
        depth = self.params.depth_spl(time)
        water_level = depth + self.initial_elevation
        degradation = (time - time[0]) * self.params.constant_rates
        elevation_change = aggradation + degradation
        index = self.start + pd.to_timedelta(time, unit='s')
        self.result = pd.DataFrame(
            data={
                "water_level": water_level,
                "depth": depth,
                "concentration": concentration,
                "aggradation": aggradation,
                "degradation": degradation,
                "elevation_change": elevation_change,
                "elevation": self.initial_elevation + elevation_change,
            },
            index=index,
        )
        self.total_aggradation = self.result.aggradation.values[-1]
        self.total_subsidence = self.result.degradation.values[-1]

    def plot(self):

        mosaic = [
            ['e', 'i'],
            ['c', 'i'],
            ['d', 'i']
        ]

        fig, ax = plt.subplot_mosaic(mosaic=mosaic, figsize=(12, 8), sharex=True, gridspec_kw={'width_ratios': [5, 2]})

        index = self.result.asfreq('10S').index
        values = self.params.depth_spl((index - self.start).total_seconds().values) + self.initial_elevation
        water_levels = pd.Series(data=values, index=index)
        sns.lineplot(data=water_levels, color="cornflowerblue", label="Tide", ax=ax['e'])
        sns.lineplot(data=self.result.elevation, color="black", ls=":", label="Land Surface", ax=ax['e'])
        ax['e'].set_ylabel(ylabel="Elevation (m)")

        sns.lineplot(data=self.result.concentration, color="saddlebrown", label="SSC", ax=ax['c'])
        ax['c'].set_ylabel(ylabel="Concentration (g/L)")

        sns.lineplot(data=self.result.aggradation * 1000, label="Aggradation", color="green", ax=ax['d'])
        sns.lineplot(data=self.result.degradation * 1000, label="Degradation", color="red", ax=ax['d'])
        sns.lineplot(
            data=self.result.elevation_change * 1000,
            label="Elevation Change",
            color='black', ls=':', ax=ax['d']
            )
        ax['d'].set_ylabel(ylabel="Height (mm)")
        ax['d'].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        data = {
            'start': self.start,
            'end': self.end,
            'period': f"{self.period.components.hours:02}H {self.period.components.minutes:02}M {self.period.components.seconds:02}S",
            'aggradation': f"{self.total_aggradation:.2e}",
            'degradation': f"{self.total_subsidence:.2e}",
            '$\Delta$elevation': f"{self.total_aggradation + self.total_subsidence:.2e}",
            **self.solve_ivp_opts
        }
        info = pd.DataFrame(data=data.values(), index=data.keys())

        for v in ['e', 'c', 'd']:
            a = ax[v]
            a.axvline(self.slack, color="black", linestyle="--")
            a.ticklabel_format(axis="y", useOffset=False)

        ax['i'].table(cellText=info.values, rowLabels=info.index, cellLoc='center', bbox=[0.25, 0.25, 0.5, 0.5])
        ax['i'].axis('off')

        ax['e'].text(x=self.slack, y=ax['e'].get_ylim()[1], s='slack', ha='center')
        plt.xticks(rotation=45)
