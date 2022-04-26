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
    flood: InundationResult = field(init=False)
    ebb: InundationResult = field(init=False)
    data: pd.DataFrame = field(init=False)
    solve_ivp_opts: dict
    aggradation_total: float = 0.0
    degradation_total: float = 0.0
    valid: bool = True

    def __post_init__(self, ssc_boundary, bulk_density, settling_rate, constant_rates):
        self.start = self.water_levels.index[0]
        self.end = self.water_levels.index[-1]
        self.logger = logger.bind(model_time=self.start)
        self.timestep = self.water_levels.index[1] - self.water_levels.index[0]
        self.period = self.end - self.start
        self.slack_time = self.water_levels.idxmax()
        self.slack_depth = self.water_levels.max() - self.initial_elevation
        while self.water_levels.shape[0] <= 3:
            freq = (self.water_levels.index.freq / 2).freqstr
            self.logger.debug(
                f"Inundation is < 3 data points. Spline interpolation requires k=3. Upsampling to {freq}."
            )
            self.water_levels = self.water_levels.asfreq(freq=freq).interpolate()
        depth = InterpolatedUnivariateSpline(x=utils.datetime2num(
            self.water_levels.index), y=self.water_levels.values - self.initial_elevation, k=3)
        self.params = Bunch(
            depth=depth,
            ssc_boundary=ssc_boundary,
            bulk_density=bulk_density,
            settling_rate=settling_rate,
            constant_rates=constant_rates,
        )

    def zero_conc(t, y, params):
        return y[0] #- 1e-4
    zero_conc.terminal = True
    zero_conc.direction = -1
    zero_conc = staticmethod(zero_conc)

    # def zero_depth(t, y, params):
    #     return params.depth(t) #- 1e-4
    # zero_depth.terminal = True
    # zero_depth.direction = -1
    # zero_depth = staticmethod(zero_depth)

    @staticmethod
    def solve_flood(t, y, params):
        concentration = y[0]

        d1dt_aggradation = params.settling_rate * concentration / params.bulk_density
        d1dt_concentration = (-(params.settling_rate * concentration) / params.depth(t) - 1 /
                              params.depth(t) * (concentration - params.ssc_boundary) * params.depth.derivative()(t))

        return [d1dt_concentration, d1dt_aggradation]

    @staticmethod
    def solve_ebb(t, y, params):
        concentration = np.sqrt(y[0])

        d1dt_aggradation = params.settling_rate * concentration / params.bulk_density
        d1dt_concentration = -(params.settling_rate * concentration) / params.depth(t)

        return [d1dt_concentration ** 2, d1dt_aggradation]

    def zero_conc2(t, y, params, mid_t):
        return y[0] #- 1e-4
    zero_conc2.terminal = True
    zero_conc2.direction = -1
    zero_conc2 = staticmethod(zero_conc2)

    # def zero_depth2(t, y, params, mid_t):
    #     return y[0] #- 1e-4
    # zero_depth2.terminal = True
    # zero_depth2.direction = -1
    # zero_depth2 = staticmethod(zero_depth2)

    @staticmethod
    def solve_odes(t, y, params, mid_t):
        concentration = y[0]

        d1dt_aggradation = params.settling_rate * concentration / params.bulk_density

        if t < mid_t:
            d1dt_concentration = (-(params.settling_rate * concentration) / params.depth(t) - 1 /
                                  params.depth(t) * (concentration - params.ssc_boundary) * params.depth.derivative()(t))
        else:
            d1dt_concentration = -(params.settling_rate * concentration) / params.depth(t)

        # if concentration + d1dt_concentration < 0:
        #     d1dt_concentration = - concentration

        return [d1dt_concentration, d1dt_aggradation]

    def integrate2(self):

        self.result = solve_ivp(
            fun=self.solve_odes,
            t_span=[utils.datetime2num(self.start), utils.datetime2num(self.end)],
            y0=[0.0, 0.0],
            # events=[self.zero_depth2],
            args=(self.params, utils.datetime2num(self.slack_time)),
            **self.solve_ivp_opts,
        )
        self.set2()

    def integrate(self):

        self.flood = solve_ivp(
            fun=self.solve_flood,
            t_span=[utils.datetime2num(self.start), utils.datetime2num(self.slack_time)],
            y0=[0.0, 0.0],
            # events=self.zero_depth,
            args=[self.params],
            **self.solve_ivp_opts,
        )
        self.validate_result(self.flood, 'flood')
        if self.flood.status == 1:
            t = np.linspace(utils.datetime2num(self.slack_time), utils.datetime2num(self.end), num=50)
            y = np.array([np.zeros(len(t)), np.repeat(a=self.flood.y[1][-1], repeats=len(t))])
            message = f'All sediment extracted on flood. {self.start}'
            self.logger.debug(message)
            self.ebb = OdeResult(t=t, y=y, status=2, message=message, success=True)
        else:
            self.ebb = solve_ivp(
                fun=self.solve_ebb,
                t_span=[utils.datetime2num(self.slack_time), utils.datetime2num(self.end)],
                y0=[self.flood.y[0][-1], self.flood.y[1][-1]],
                # events=self.zero_depth,
                args=[self.params],
                **self.solve_ivp_opts,
            )
            self.validate_result(self.ebb, 'ebb')
        self.set()

    def validate_result(self, result, limb) -> None:
        if result.success is False:
            self.valid = False
            self.logger.warning(
                f"Integration of {limb} failed using {self.solve_ivp_opts['method']} with message: {result.message}")
        if (result.y[1] < 0.0).any():
            self.valid = False
            self.logger.warning(f"Solution for {limb} contains negative aggradations.")
        if result.y[1][-1] > self.slack_depth * self.params.ssc_boundary:
            self.valid = False
            self.logger.warning("More sediment extracted than possible on {limb}.")

    def set(self):
        time = np.append(self.flood.t, self.ebb.t[1:])
        concentration = np.append(self.flood.y[0], self.ebb.y[0][1:])
        aggradation = np.append(self.flood.y[1], self.ebb.y[1][1:])
        if utils.num2datetime(time[-1]) != self.end:
            time = np.append(time, utils.datetime2num(self.end))
            concentration = np.append(concentration, 0.0)
            aggradation = np.append(aggradation, aggradation[-1])
        index = utils.num2datetime(time)
        depth = self.params.depth(time)
        water_level = depth + self.initial_elevation
        degradation = (time - time[0]) * self.params.constant_rates
        elevation_change = aggradation + degradation
        self.data = pd.DataFrame(
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
        self.aggradation_total = self.data.aggradation.values[-1]
        self.degradation_total = self.data.degradation.values[-1]

    def set2(self):
        time = self.result.t
        concentration = self.result.y[0]
        aggradation = self.result.y[1]
        # if utils.num2datetime(self.result.t[-1]) != self.end:
        #     time = np.append(time, utils.datetime2num(self.end))
        #     concentration = np.append(concentration, 0.0)
        #     aggradation = np.append(aggradation, aggradation[-1])
        index = utils.num2datetime(time)
        depth = self.params.depth(time)
        water_level = depth + self.initial_elevation
        degradation = (time - time[0]) * self.params.constant_rates
        elevation_change = aggradation + degradation
        self.data = pd.DataFrame(
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
        self.aggradation_total = self.data.aggradation.values[-1]
        self.degradation_total = self.data.degradation.values[-1]

    def plot(self):

        mosaic = [
            ['e', 'i'],
            ['c', 'i'],
            ['d', 'i']
        ]

        fig, ax = plt.subplot_mosaic(mosaic=mosaic, figsize=(12, 8), sharex=True, gridspec_kw={'width_ratios': [5, 2]})

        index = self.data.asfreq('10S').index
        values = self.params.depth(utils.datetime2num(index)) + self.initial_elevation
        water_levels = pd.Series(data=values, index=index)
        sns.lineplot(data=water_levels, color="cornflowerblue", label="Tide", ax=ax['e'])
        sns.lineplot(data=self.data.elevation, color="black", ls=":", label="Land Surface", ax=ax['e'])
        ax['e'].set_ylabel(ylabel="Elevation (m)")

        sns.lineplot(data=self.data.concentration, color="saddlebrown", label="SSC", ax=ax['c'])
        ax['c'].set_ylabel(ylabel="Concentration (g/L)")

        sns.lineplot(data=self.data.aggradation * 1000, label="Aggradation", color="green", ax=ax['d'])
        sns.lineplot(data=self.data.degradation * 1000, label="Degradation", color="red", ax=ax['d'])
        sns.lineplot(data=self.data.elevation_change * 1000,
                     label="Elevation Change", color='black', ls=':', ax=ax['d'])
        ax['d'].set_ylabel(ylabel="Height (mm)")
        ax['d'].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        data = {
            'start': self.start,
            'end': self.end,
            'period': f"{self.period.components.hours:02}H {self.period.components.minutes:02}M {self.period.components.seconds:02}S",
            'aggradation': f"{self.aggradation_total:.2e}",
            'degradation': f"{self.degradation_total:.2e}",
            '$\Delta$elevation': f"{self.aggradation_total + self.degradation_total:.2e}",
            # 'msg': '\n'.join(wrap(self.result.message, 20)),
            **self.solve_ivp_opts
        }
        info = pd.DataFrame(data=data.values(), index=data.keys())

        # int_end = utils.num2datetime(self.result.t[-1])

        for v in ['e', 'c', 'd']:
            a = ax[v]
            a.axvline(self.slack_time, color="black", linestyle="--")
            # a.axvline(int_end, color='black', ls=':')
            a.ticklabel_format(axis="y", useOffset=False)

        ax['i'].table(cellText=info.values, rowLabels=info.index, cellLoc='center', bbox=[0.25, 0.25, 0.5, 0.5])
        ax['i'].axis('off')

        ax['e'].text(x=self.slack_time, y=ax['e'].get_ylim()[1], s='slack', ha='center')
        # ax['e'].text(x=int_end, y=ax['e'].get_ylim()[1], s='end', ha='center')
        plt.xticks(rotation=45)
