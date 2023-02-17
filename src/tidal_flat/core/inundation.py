from dataclasses import dataclass, field

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
    start: pd.Timedelta
    mid: pd.Timedelta
    end: pd.Timedelta
    time_ref: pd.Timestamp
    cycle_n: int
    hydroperiod: pd.Timedelta = field(init=False)
    solve_ivp_kwargs: dict = field(init=False, default_factory=dict)
    result: pd.DataFrame | None = field(init=False, default=None)
    valid: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.logger = logger.bind(model_time=self.start)
        self.hydroperiod = self.end - self.start
        self.solve_ivp_kwargs = {
            'method': 'RK45',
            'dense_output': False,
            'first_step': None,
            'max_step': np.inf,
            'rtol': 1e-3,
            'atol': 1e-6,
        }

    @property
    def slack(self) -> pd.Series:
        return pd.Series(
            data=[self.mid, self.depth(self.mid.total_seconds()).item()],
            index=['time', 'depth'],
        )

    @property
    def flood(self) -> pd.Series:
        return pd.Series(
            data=[self.start, self.slack.time, self.slack.time - self.start],
            index=['start', 'end', 'period'],
            name='flood',
        )

    @property
    def ebb(self) -> pd.Series:
        return pd.Series(
            data=[self.slack.time, self.end, self.end - self.slack.time],
            index=['start', 'end', 'period'],
            name='ebb',
        )

    @property
    def aggradation(self) -> float | None:
        return self.result.aggradation.iat[-1] if self.result is not None else None

    @property
    def summary(self) -> pd.Series:
        return pd.Series(
            data={
                'start': self.start + self.time_ref,
                'end': self.end + self.time_ref,
                'hydroperiod': self.hydroperiod,
                'aggradation': self.aggradation,
                'max_depth': self.slack.depth,
                'mean_depth': self.depth(
                    np.linspace(self.start.total_seconds(), self.end.total_seconds(), 1000)
                ).mean(),
                'valid': self.valid,
            },
            name=self.cycle_n,
        )

    @staticmethod
    def solve_flood(
        t: float, y: np.ndarray, depth: PchipInterpolator, ssc: float, bulk_density: float, settling_rate: float
    ) -> list[float]:
        concentration = y[0]

        d1dt_aggradation = settling_rate * concentration / bulk_density
        d1dt_concentration = -(settling_rate * concentration) / depth(t) - 1 / depth(t) * (
            concentration - ssc
        ) * depth.derivative()(t)

        return [d1dt_concentration, d1dt_aggradation]

    @staticmethod
    def solve_ebb(
        t: float, y: np.ndarray, depth: PchipInterpolator, bulk_density: float, settling_rate: float
    ) -> list[float]:
        concentration = y[0]

        d1dt_aggradation = settling_rate * concentration / bulk_density
        d1dt_concentration = -(settling_rate * concentration) / depth(t)

        return [d1dt_concentration, d1dt_aggradation]

    def solve(self, ssc: float, bulk_density: float, settling_rate: float) -> None:
        self.logger.trace('Integrating flood limb.')
        params = Bunch(depth=self.depth, ssc=ssc, bulk_density=bulk_density, settling_rate=settling_rate)
        flood = solve_ivp(
            fun=self.solve_flood,
            t_span=[self.flood.start.total_seconds(), self.flood.end.total_seconds()],
            y0=(0.0, 0.0),
            args=[params.depth, params.ssc, params.bulk_density, params.settling_rate],
            **self.solve_ivp_kwargs,
        )
        flood.params = params
        self.validate(flood)

        self.logger.trace('Integrating ebb limb.')
        ebb = solve_ivp(
            fun=self.solve_ebb,
            t_span=[self.ebb.start.total_seconds(), self.ebb.end.total_seconds()],
            y0=(flood.y[0][-1], flood.y[1][-1]),
            args=[params.depth, params.bulk_density, params.settling_rate],
            **self.solve_ivp_kwargs,
        )
        ebb.params = params
        self.validate(ebb)
        if flood.valid and ebb.valid:
            self.valid = True

        self.concat_results(flood, ebb)

    def validate(self, result: Bunch) -> None:
        result.valid = True
        if result.success is False:
            result.valid = False
            self.logger.warning(result.message)
        if (result.y[1] < 0.0).any():
            result.valid = False
            self.logger.warning('Solution contains negative aggradations.')
        if result.y[1][-1] > self.slack.depth * result.params.ssc:
            result.valid = False
            self.logger.warning('More sediment extracted than possible.')

    def concat_results(self, flood: Bunch, ebb: Bunch) -> None:
        time = np.append(flood.t, ebb.t[1:])
        concentration = np.append(flood.y[0], ebb.y[0, 1:])
        aggradation = np.append(flood.y[1], ebb.y[1, 1:])
        depth = self.depth(time)
        index = pd.to_timedelta(time, unit='s')
        self.result = pd.DataFrame(
            data={
                'depth': depth,
                'concentration': concentration,
                'aggradation': aggradation,
            },
            index=index,
        )

    def plot(self) -> None:
        if self.result:
            mosaic = [['e', 'i'], ['c', 'i'], ['d', 'i']]

            fig, ax = plt.subplot_mosaic(
                mosaic=mosaic, figsize=(12, 8), sharex=True, gridspec_kw={'width_ratios': [5, 2]}
            )

            # index = pd.date_range(self.flood.start, self.ebb.end, freq="10S")
            index = pd.timedelta_range(self.flood.start, self.ebb.end, freq='10S')
            # values = self.depth(index.astype(int) / 10**9)
            values = self.depth(index.total_seconds())
            # values = self.params.depth_spl((index - self.start).total_seconds().values) + self.initial_elevation
            # water_levels = pd.Series(data=values, index=index)
            sns.lineplot(x=index, y=values, color='cornflowerblue', label='Tide', ax=ax['e'])
            # sns.lineplot(data=self.tides.loc[self.start:self.end] - self.initial_elevation, color="red", ax=ax["e"])
            sns.lineplot(data=self.result.aggradation, color='black', ls=':', label='Land Surface', ax=ax['e'])
            ax['e'].set_ylabel(ylabel='Depth (m)')

            sns.lineplot(data=self.result.concentration, color='saddlebrown', label='SSC', ax=ax['c'])
            ax['c'].set_ylabel(ylabel='Concentration (g/L)')

            sns.lineplot(data=self.result.aggradation * 1000, label='Aggradation', color='green', ax=ax['d'])
            ax['d'].set_ylabel(ylabel='Height (mm)')
            # ax["d"].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

            data = {
                'start': self.start.round('S'),
                'end': self.end.round('S'),
                'period': (
                    f'{self.hydroperiod.components.hours:02}H {self.hydroperiod.components.minutes:02}M'
                    f' {self.hydroperiod.components.seconds:02}S'
                ),
                # "aggradation": f"{self.aggradation:.2e}",
                # "degradation": f"{self.subsidence:.2e}",
                # "$\Delta$elevation": f"{self.aggradation + self.subsidence:.2e}",
                **self.solve_ivp_kwargs,
            }
            info = pd.DataFrame.from_dict(data, orient='index')

            for v in ['e', 'c', 'd']:
                a = ax[v]
                a.axvline(self.slack.time / pd.Timedelta(1, unit='ns'), color='black', linestyle='--')
                a.ticklabel_format(axis='y', useOffset=False)

            ax['i'].table(cellText=info.values, rowLabels=info.index, cellLoc='center', bbox=[0.25, 0.25, 0.5, 0.5])
            ax['i'].axis('off')

            ax['e'].text(
                x=self.slack.time / pd.Timedelta(1, unit='ns'), y=ax['e'].get_ylim()[1], s='slack', ha='center'
            )
            plt.xticks(rotation=45)
