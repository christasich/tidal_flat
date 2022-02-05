import time
from collections import namedtuple
from dataclasses import InitVar, dataclass, field

import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from scipy.interpolate import InterpolatedUnivariateSpline
from tqdm.notebook import tqdm

from . import constants


class InundationResult(OdeResult):
    pass


@dataclass
class Model:
    tides: pd.Series
    initial_elevation: float | int
    ssc: InitVar[float | int | np.ndarray]
    grain_diameter: float
    grain_density: float
    bulk_density: float
    organic_rate: float = 0.0
    compaction_rate: float = 0.0
    subsidence_rate: float = 0.0
    elevation: float = field(init=False)
    aggradation_total: float = field(init=False)
    degradation_total: float = field(init=False)
    inundations: dict = field(init=False, default_factory=dict)
    results: list = field(init=False, default_factory=list)
    verbose: bool = False
    initial_time: pd.Timestamp = field(init=False)
    current_time: pd.Timestamp = field(init=False)
    timespan: pd.Timedelta = field(init=False)
    timestep: pd.Timedelta = field(init=False)

    def __post_init__(self, ssc: float | int | np.ndarray) -> None:

        self.elevation = self.initial_elevation
        self.aggradation_total = 0.0
        self.degradation_total = 0.0
        self.initial_time = self.current_time = self.tides.index[0]
        self.timespan = self.tides.index[-1] - self.tides.index[0]
        self.timestep = pd.Timedelta(self.tides.index.freq)

        # repeat ssc for each month if given a single number
        if isinstance(ssc, (float, int)):
            ssc = np.repeat(ssc, 12)

        # make a daily ssc index to match tides
        # set value at midpoint of each month
        # add one month on each side for better interpolation
        ssc_index = pd.date_range(
            start=self.tides.index[0] - pd.DateOffset(months=1),
            end=self.tides.index[-1] + pd.DateOffset(months=1),
            freq='MS',
        ) + pd.DateOffset(days=14)

        # make a series from ssc array and new index then upsample to 1D and interpolate.
        ssc_series = pd.Series(data=ssc[ssc_index.month - 1], index=ssc_index).asfreq('1D').interpolate()

        # remove trailing and lead months to match original index
        self.ssc = ssc_series.loc[self.tides.index[0] : self.tides.index[-1]]

    @staticmethod
    def stokes_settling(
        grain_diameter: float,
        grain_density: float,
        fluid_density: float = constants.WATER_DENSITY,
        fluid_viscosity: float = constants.WATER_VISCOSITY,
        gravity: float = constants.GRAVITY,
    ) -> float:
        settling_rate = (
            (2 / 9 * (grain_density - fluid_density) / fluid_viscosity) * gravity * (grain_diameter / 2) ** 2
        )
        return settling_rate

    @property
    def settling_rate(self) -> float:
        return self.stokes_settling(grain_diameter=self.grain_diameter, grain_density=self.grain_density)

    @property
    def combined_linear_rates(self) -> float:
        yearly = self.organic_rate + self.compaction_rate + self.subsidence_rate
        return yearly / pd.Timedelta('365.25D').total_seconds()

    def make_subset(self):

        window = pd.Timedelta('1D')
        end = self.current_time + window

        subset = self.tides.loc[self.current_time : end].to_frame(name='tide_elev')
        subset['land_elev'] = (
            self.land_elev + (subset.index - subset.index[0]).total_seconds().values * self.linear_rate
        )
        num_crossings = len(np.where(np.diff(np.signbit(subset.tide_elev - subset.land_elev)))[0])

        count = 0
        while num_crossings < 2:
            if subset.index[-1] == self.tides.index[-1] and num_crossings == 1:
                return subset
            elif subset.index[-1] == self.tides.index[-1] and num_crossings == 0:
                return subset
            else:
                end = end + window
                subset = self.tides.loc[self.current_time : end].to_frame(name='tide_elev')
                subset['land_elev'] = (
                    self.land_elev + (subset.index - subset.index[0]).total_seconds().values * self.linear_rate
                )
                num_crossings = len(np.where(np.diff(np.signbit(subset.tide_elev - subset.land_elev)))[0])
            count += 1
            if count > 7:
                window = pd.Timedelta('7D')
            elif count > 7 + 2:
                window = pd.Timedelta('30D')
        return subset

    def find_inundation(self):
        # status:
        #  -2 = initial tide above platform
        #  -1 = skipping inundation since len < 3
        #   0 = full inundation cycle
        #   1 = end of tidal data
        subset = self.make_subset()
        if subset.index[-1] == self.tides.index[-1]:
            return [subset, None, 1]

        if self.current_time == self.tides.index[0] and subset.tide_elev.loc[self.current_time] > self.land_elev:
            t_start = self.current_time
        else:
            t_start = (subset.tide_elev > subset.land_elev).idxmax()

        t_end = (subset.loc[t_start:].tide_elev > subset.loc[t_start:].land_elev).idxmin()
        assert t_end > t_start

        if subset.loc[t_start:t_end].shape[0] < 3:
            return [subset.loc[self.current_time : t_end], None, -1]
        init_elev = subset.land_elev.loc[t_start]
        inundation = Inundation(
            tides=subset.loc[t_start:t_end],
            init_elev=init_elev,
            conc_bound=self.ssc[t_start.round('1D')],
            settle_rate=self.settling_rate,
            bulk_dens=self.bulk_density,
            linear_rate=self.linear_rate,
        )
        if self.current_time == self.tides.index[0] and subset.tide_elev.loc[self.current_time] > self.land_elev:
            return [None, inundation, -2]
        else:
            return [subset.loc[self.current_time : t_start - self.timestep], inundation, 0]

    def step(self):
        subset_before, inundation, status = self.find_inundation()
        if status == 0 or status == -2:
            self.inundations[inundation.t_start.strftime('%Y-%m-%d %X')] = inundation
            inundation.integrate()
        self.update(subset_before, inundation, status)

    def run(self):
        self._initialize()
        while self.current_time < self.tides.index[-1]:
            self.step()
            self.pbar.n = (self.current_time - self.tides.index[0]).ceil('D').days
            self.pbar.set_postfix({'Date': self.current_time.strftime('%Y-%m-%d')})
        self._unitialize()

    def _initialize(self):

        self.runtime = time.perf_counter()
        postfix = {'Date': self.tides.index[0].strftime('%Y-%m-%d')}
        self.pbar = tqdm(total=self.timespan.ceil('D').days, unit='Day', position=0, leave=True, postfix=postfix)

    def _unitialize(self):
        self.runtime = time.perf_counter() - self.runtime
        self.pbar.close()
        if self.verbose is True:
            self.print_results()

    def update(self, subset, inundation, status):
        if status == 0:
            self.results.append(subset)
            self.degradation_total = self.degradation_total + (subset.land_elev.values[-1] - subset.land_elev.values[0])
            self.degradation_total = self.degradation_total + inundation.degr_total
            self.aggradation_total = self.aggradation_total + inundation.aggr_total
            self.results.append(inundation.df[['tide_elev', 'land_elev']])
            self.land_elev = inundation.result.y[2][-1]
            self.current_time = inundation.t_end + self.timestep
        elif status == -1:
            self.results.append(subset)
            self.degradation_total = self.degradation_total + (subset.land_elev.values[-1] - subset.land_elev.values[0])
            self.land_elev = subset.land_elev.values[-1]
            self.current_time = subset.index[-1] + self.timestep
        elif status == 1:
            self.results.append(subset)
            self.degradation_total = self.degradation_total + (subset.land_elev.values[-1] - subset.land_elev.values[0])
            self.results = pd.concat(self.results)
            self.land_elev = subset.land_elev.values[-1]
            self.current_time = subset.index[-1] + self.timestep
        elif status == -2:
            self.degradation_total = self.degradation_total + inundation.degr_total
            self.aggradation_total = self.aggradation_total + inundation.aggr_total
            self.results.append(inundation.df[['tide_elev', 'land_elev']])
            self.land_elev = inundation.result.y[2][-1]
            self.current_time = inundation.t_end + self.timestep

    def print_results(self):
        years = self.timespan / pd.Timedelta('365.25D')
        print('{:<25} {:>10} {:>10} {:>5}'.format('', 'Mean Yearly', 'Total', 'Unit'))
        print('-' * 55)
        print('{:<25} {:>10} {:>10.3f} {:>5}'.format('Starting elevation: ', '', self.results.land_elev.iat[0], 'm'))
        print('{:<25} {:>10} {:>10.3f} {:>5}'.format('Final elevation: ', '', self.results.land_elev.iat[-1], 'm'))
        print(
            '{:<25} {:>10.3f} {:>10.3f} {:>5}'.format(
                'Elevation change: ',
                (self.results.land_elev.iat[-1] - self.results.land_elev.iat[0]) * 100 / years,
                (self.results.land_elev.iat[-1] - self.results.land_elev.iat[0]) * 100,
                'cm',
            )
        )
        print('-' * 55)
        print(
            '{:<25} {:>10.3f} {:>10.3f} {:>5}'.format(
                'Aggradation: ', self.aggradation_total * 100 / years, self.aggradation_total * 100, 'cm'
            )
        )
        print(
            '{:<25} {:>10.3f} {:>10.3f} {:>5}'.format(
                'Degradation: ', self.degradation_total * 100 / years, self.degradation_total * 100, 'cm'
            )
        )
        print('-' * 55)
        print('{:<25} {:>25}'.format('Runtime: ', time.strftime('%M min %S s', time.gmtime(self.runtime))))

    def plot(self, unit="H"):
        data = self.results.resample(unit).mean()

        fig, ax1 = plt.subplots(figsize=(15, 5), constrained_layout=True)
        ax2 = ax1.twinx()
        sns.lineplot(
            ax=ax1,
            x=data.index,
            y=data.tide_elev,
            alpha=0.6,
            color='cornflowerblue',
            label='Tide Elevation',
            legend=False,
        )
        sns.lineplot(
            ax=ax2,
            x=data.index,
            y=data.land_elev - self.results.land_elev.iat[0],
            color='forestgreen',
            label='Land Elevation',
            legend=False,
        )
        ax1.set(
            xlim=(self.results.index[0], self.results.index[-1]),
            ylim=(self.results.land_elev.min(), self.results.tide_elev.max()),
            xlabel='Year',
            ylabel='Elevation (m)',
        )
        ax2.set(
            xlim=(self.results.index[0], self.results.index[-1]),
            ylim=(
                self.results.land_elev.min() - self.results.land_elev.iat[0],
                self.results.tide_elev.max() - self.results.land_elev.iat[0],
            ),
            ylabel=r'$\Delta$ Elevation (m)',
        )
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2)
        return (fig, [ax1, ax2])


@dataclass
class Inundation:
    tides: pd.DataFrame
    init_elev: float
    conc_bound: InitVar[float]
    settle_rate: InitVar[float]
    bulk_dens: InitVar[float]
    linear_rate: InitVar[float]
    result: InundationResult = field(default=None, init=False)
    df: pd.DataFrame = field(default=None, init=False)
    aggr_total: float = field(default=None, init=False)
    degr_total: float = field(default=None, init=False)

    def __post_init__(self, conc_bound, settle_rate, bulk_dens, linear_rate):
        self._timestep = self.tides.index[1] - self.tides.index[0]
        self.t_start = self.tides.index[0]
        self.t_end = self.tides.index[-1]
        self.t_slack = self.tides.tide_elev.idxmax()
        self.tide_elev_slack = self.tides.tide_elev.max()
        self.period = self.t_end - self.t_start
        tide_elev_func = InterpolatedUnivariateSpline(
            x=self.tides.index.astype(np.int64) / 10**9, y=self.tides.tide_elev.values, k=3
        )
        params = namedtuple(
            'params',
            ['tide_elev_func', 'conc_bound', 'settle_rate', 'bulk_dens', 'linear_rate'],
        )
        self.params = params(
            tide_elev_func=tide_elev_func,
            conc_bound=conc_bound,
            settle_rate=settle_rate,
            bulk_dens=bulk_dens,
            linear_rate=linear_rate,
        )

    def _set_df(self):
        time = pd.to_datetime(self.result.t, unit="s")
        tide_elev = self.result.y[0]
        conc = self.result.y[1]
        land_elev = self.result.y[2]
        aggr = self.result.y[3]
        aggr_max = self.result.y[4]
        degr = self.result.y[5]
        df = pd.DataFrame(
            data={
                'tide_elev': tide_elev,
                'land_elev': land_elev,
                'conc': conc,
                'aggr': aggr,
                'aggr_max': aggr_max,
                'degr': degr,
            },
            index=time,
        )
        if self.result.status == 1:
            solver_end_time = time[-1]
            next_pos = solver_end_time.ceil(self.tides.index.freq)
            solver_end_diff = next_pos - solver_end_time
            small_degr = solver_end_diff.total_seconds() * self.params.linear_rate
            df2 = self.tides.loc[next_pos:].copy()
            df2['land_elev'] = df.land_elev.values[-1]
            df2['conc'] = 0.0
            df2['aggr'] = df.aggr.values[-1]
            df2['aggr_max'] = df.aggr_max.values[-1]
            df2['degr'] = (
                df.degr.values[-1]
                + small_degr
                + self.params.linear_rate * (df2.index - df2.index[0]).total_seconds().values
            )
            df = pd.concat([df, df2])

        self.degr_total = df.degr.values[-1]
        self.df = df

    @staticmethod
    def solve_odes(t, y, params):
        tide_elev = y[0]
        conc = y[1]
        land_elev = y[2]
        depth = tide_elev - land_elev

        d1dt_tide_elev = params.tide_elev_func.derivative()(t)
        d1dt_aggr = params.settle_rate * conc / params.bulk_dens
        d1dt_degr = params.linear_rate
        d1dt_land_elev = d1dt_aggr + d1dt_degr
        d1dt_depth = d1dt_tide_elev - d1dt_land_elev
        if d1dt_depth > 0:
            d1dt_conc = -(params.settle_rate * conc) / depth - 1 / depth * (conc - params.conc_bound) * d1dt_depth
            d1dt_aggr_max = params.conc_bound * d1dt_depth / params.bulk_dens
        else:
            d1dt_conc = -(params.settle_rate * conc) / depth
            d1dt_aggr_max = 0.0

        return [d1dt_tide_elev, d1dt_conc, d1dt_land_elev, d1dt_aggr, d1dt_aggr_max, d1dt_degr]  # 0  # 1  # 2  # 3  # 4

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

    def integrate(self, method='DOP853', dense_output=True):

        self.result = solve_ivp(
            fun=self.solve_odes,
            t_span=[self.t_start.timestamp(), self.t_end.timestamp()],
            y0=[self.tides.tide_elev.values[0], 0.0, self.init_elev, 0.0, 0.0, 0.0],
            method=method,
            events=(self.zero_conc, self.zero_depth),
            dense_output=dense_output,
            atol=(1e-6, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8),
            args=[self.params],
        )

        self.aggr_total = self.result.y[3][-1]
        self._set_df()
        self._validate_result()

    def _validate_result(self):
        if self.result.success is False:
            logger.warning(f'{self.t_start} | Integration failed!\nparams={self.params}')
        if (self.df.aggr < 0.0).any():
            logger.warning(f'{self.t_start} | Negative aggradation detected!\n{self.df.loc[self.df.aggr < 0]}')
        if (self.df.aggr_max < self.df.aggr).any():
            logger.warning(f'{self.t_start} | Overextraction detected!\n{self.df.loc[self.df.aggr_max < self.df.aggr]}')

    def plot(self):

        fig, axs = plt.subplots(nrows=4, ncols=1, tight_layout=True)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        fig.suptitle(f'Inundation at {self.tides.index[0]}', fontsize=16)

        aggr_max_mod_diff = self.df.aggr_max - self.df.aggr

        sns.lineplot(ax=axs[0], data=self.df.tide_elev, color='cornflowerblue', label='Tide')
        sns.lineplot(ax=axs[0], data=self.df.land_elev, color='forestgreen', label='Land')
        axs[0].set(xlabel='', ylabel='Elevation (m)', xticklabels=[])

        sns.lineplot(ax=axs[1], data=self.df.conc, color='saddlebrown')
        axs[1].set(xlabel='', ylabel='Concentration (g/L)', xticklabels=[])

        sns.lineplot(ax=axs[2], data=self.df.aggr_max, color='red', label='Max')
        sns.scatterplot(ax=axs[2], data=self.df.aggr, label='Modeled')
        axs[2].set(xlabel='', ylabel='Aggradation (m)', xticklabels=[])
        axs[2].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        sns.lineplot(ax=axs[3], data=aggr_max_mod_diff, color='black', linestyle=':')
        axs[3].set(ylabel='Difference (m)\nMax - Modeled', xlabel='Time (min)')
        axs[3].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        axs[3].fill_between(
            x=self.df.index, y1=aggr_max_mod_diff, where=aggr_max_mod_diff >= 0.0, color='forestgreen', alpha=0.3
        )
        axs[3].fill_between(x=self.df.index, y1=aggr_max_mod_diff, where=aggr_max_mod_diff < 0, color='red', alpha=0.3)
        for ax in axs:
            ax.axvline(self.t_slack, color='black', linestyle='--')
            ax.ticklabel_format(axis='y', useOffset=False)