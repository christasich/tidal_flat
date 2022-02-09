import time
from dataclasses import InitVar, dataclass, field

import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import root
from sklearn.utils import Bunch
from tqdm.notebook import tqdm

from . import constants
from .utils import calculate_rate_vector, datetime2num, find_roots, num2datetime, stokes_settling


class InundationResult(OdeResult):
    pass


@dataclass
class Model:
    tides: InitVar[pd.Series]
    elevation_start: float | int
    ssc: InitVar[float | int | np.ndarray]
    settling_rate: InitVar[float]
    bulk_density: InitVar[float]
    organic_rate: InitVar[float] = 0.0
    compaction_rate: InitVar[float] = 0.0
    subsidence_rate: InitVar[float] = 0.0
    data: pd.DataFrame = field(init=False)
    aggradation_total: float = field(init=False)
    degradation_total: float = field(init=False)
    inundations: list = field(init=False, default_factory=list)
    results: list = field(init=False)
    verbose: bool = False
    now: pd.Timestamp = field(init=False)
    start: pd.Timestamp = field(init=False)
    end: pd.Timestamp = field(init=False)
    timespan: pd.Timedelta = field(init=False)
    timestep: pd.Timedelta = field(init=False)

    @property
    def constant_rates(self) -> float:
        return (self.params.organic_rate + self.params.compaction_rate + self.params.subsidence_rate) / constants.YEAR

    @property
    def elevation_now(self) -> float:
        return self.data.elevation.at[self.now]

    @property
    def water_level_now(self) -> float:
        return self.data.water_level.at[self.now]

    def __post_init__(
        self,
        tides,
        ssc: float | int | np.ndarray,
        settling_rate: float,
        bulk_density: float,
        organic_rate: float,
        compaction_rate: float,
        subsidence_rate: float,
    ) -> None:
        self.params = Bunch(
            ssc=ssc,
            settling_rate=settling_rate,
            bulk_density=bulk_density,
            organic_rate=organic_rate,
            compaction_rate=compaction_rate,
            subsidence_rate=subsidence_rate,
        )
        self.data = tides.to_frame('water_level')
        self.data['elevation'] = self.elevation_start + calculate_rate_vector(
            t=self.data.index, rate=self.constant_rates
        )
        self.aggradation_total = 0.0
        self.degradation_total = 0.0
        self.start = self.now = self.data.index[0]
        self.end = self.data.index[-1]
        self.timespan = self.data.index[-1] - self.data.index[0]
        self.timestep = pd.Timedelta(tides.index.freq)
        self.results = [{'index': self.start, 'elevation': self.elevation_start}]

        # repeat ssc for each month if given a single number
        if isinstance(ssc, (float, int)):
            ssc = np.repeat(ssc, 12)

        # make a daily ssc index to match tides
        # set value at midpoint of each month
        # add one month on each side for better interpolation
        ssc_index = pd.date_range(
            start=self.data.index[0] - pd.DateOffset(months=1),
            end=self.data.index[-1] + pd.DateOffset(months=1),
            freq='MS',
        ) + pd.DateOffset(days=14)

        # make a series from ssc array and new index then upsample to 1D and interpolate.
        ssc_series = pd.Series(data=ssc[ssc_index.month - 1], index=ssc_index).asfreq('1D').interpolate()

        # remove trailing and lead months to match original index
        self.ssc = ssc_series.loc[self.data.index[0] : self.data.index[-1]]

    def _initialize(self):
        self.runtime = time.perf_counter()
        logger.info('Initializing model.')
        postfix = {'Date': self.data.index[0].strftime('%Y-%m-%d')}
        self.pbar = tqdm(total=self.timespan.ceil('D').days, unit='Day', position=0, leave=True, postfix=postfix)
        # self.update_results(index=self.start, elevation=self.elevation_start)

        if self.water_level_now > self.elevation_now:
            logger.info('Tide starts above platform. Skipping first inundation.')
            below_idx = (self.data.elevation > self.data.water_level).idxmax()
            self.now = below_idx
            self.update_results(index=self.now, elevation=self.elevation_now)

    # def step(self):
    #     subset_before, inundation, status = self.next_inundation()
    #     if status == 0 or status == -2:
    #         self.inundations[inundation.t_start.strftime('%Y-%m-%d %X')] = inundation
    #         inundation.integrate()
    #     self.update(subset_before, inundation, status)

    # def run(self):
    #     self._initialize()
    #     while self.now < self.data.index[-1]:
    #         self.step()
    #         self.pbar.n = (self.now - self.data.index[0]).ceil('D').days
    #         self.pbar.set_postfix({'Date': self.now.strftime('%Y-%m-%d')})
    #     self._unitialize()

    def update_results(self, index, elevation):
        self.results.append({'index': index, 'elevation': elevation})

    def find_next_inundation(self):

        n = pd.Timedelta('1H')
        end = (self.data.water_level.loc[self.now :] > self.data.elevation.loc[self.now :]).idxmax() + n

        subset = self.data.loc[self.now : end]
        roots = find_roots(a=subset.water_level, b=subset.elevation)

        while len(roots) < 2:
            if subset.index[-1] == self.end and len(roots) == 1:
                return subset
            elif subset.index[-1] == self.end and len(roots) == 0:
                return None
            else:
                n = n * 1.1
                end = end + n
                subset = self.data.loc[self.now : end]
                roots = find_roots(a=subset.water_level, b=subset.elevation)
        return subset.iloc[roots[0] + 1 : roots[1] + 2]

    def step(self):
        subset = self.find_next_inundation()
        if subset is None:
            logger.info('No more inundations to process. Exiting.')
            self.now = self.end
            return
        else:
            inundation = Inundation(
                tides=subset.water_level,
                elevation_start=subset.elevation.iat[0],
                ssc_boundary=self.params.ssc[self.now.month - 1],
                bulk_density=self.params.bulk_density,
                settling_rate=self.params.settling_rate,
                constant_rates=self.constant_rates,
            )
            self.update_results(index=inundation.start, elevation=inundation.elevation_start)
            self.inundations.append(inundation)
        try:
            inundation.integrate()
        except:
            logger.exception('Problem with integration!')
        else:
            self.update_results(index=inundation.df.index[-1], elevation=inundation.df.elevation.iat[-1])
            self.now = inundation.end
            elevation_change = inundation.df.elevation.at[self.now] - self.data.elevation.at[self.now]
            self.data.elevation.loc[self.now :] = self.data.elevation[self.now :].values + elevation_change

    def run(self):
        self._initialize()
        while self.now < self.end:
            self.step()
            self.pbar.n = (self.now - self.start).ceil('D').days
            self.pbar.set_postfix({'Date': self.now.strftime('%Y-%m-%d')})
        self._unitialize()

    def _unitialize(self):
        self.results = pd.DataFrame.from_records(data=self.results, index='index').squeeze()
        self.runtime = time.perf_counter() - self.runtime
        self.pbar.close()
        if self.verbose is True:
            self.print_results()
        # if self.now == self.end and subset.water_level.loc[self.now] > self.elevation:
        #     t_start = self.now
        # else:
        #     t_start = (subset.tide_elev > subset.land_elev).idxmax()

        # t_end = (subset.loc[t_start:].tide_elev > subset.loc[t_start:].land_elev).idxmin()
        # assert t_end > t_start

        # if subset.loc[t_start:t_end].shape[0] < 3:
        #     return [subset.loc[self.now : t_end], None, -1]
        # init_elev = subset.land_elev.loc[t_start]
        # inundation = Inundation(
        #     tides=subset.loc[t_start:t_end],
        #     elevation_start=init_elev,
        #     ssc=self.ssc[t_start.round('1D')],
        #     settling_rate=self.settling_rate,
        #     bulk_density=self.bulk_density,
        #     constant_rates=self.constant_rates,
        # )

        # return [subset.loc[self.now : t_start - self.timestep], inundation, 0]

    # def next_inundation(self):
    #     # status:
    #     #  -1 = skipping inundation since len < 3
    #     #   0 = full inundation cycle
    #     #   1 = end of tidal data
    #     subset = self.make_subset()
    #     if subset.index[-1] == self.end:
    #         return [subset, None, 1]

    #     if self.now == self.end and subset.water_level.loc[self.now] > self.elevation:
    #         t_start = self.now
    #     else:
    #         t_start = (subset.tide_elev > subset.land_elev).idxmax()

    #     t_end = (subset.loc[t_start:].tide_elev > subset.loc[t_start:].land_elev).idxmin()
    #     assert t_end > t_start

    #     if subset.loc[t_start:t_end].shape[0] < 3:
    #         return [subset.loc[self.now : t_end], None, -1]
    #     init_elev = subset.land_elev.loc[t_start]
    #     inundation = Inundation(
    #         tides=subset.loc[t_start:t_end],
    #         init_elev=init_elev,
    #         conc_bound=self.ssc[t_start.round('1D')],
    #         settle_rate=self.settling_rate,
    #         bulk_dens=self.bulk_density,
    #         constant_rates=self.constant_rates,
    #     )

    #     return [subset.loc[self.now : t_start - self.timestep], inundation, 0]

    # def update(self, subset, inundation, status):
    #     if status == 0:
    #         self.results.append(subset)
    #         self.degradation_total = self.degradation_total + (subset.land_elev.values[-1] - subset.land_elev.values[0])
    #         self.degradation_total = self.degradation_total + inundation.degr_total
    #         self.aggradation_total = self.aggradation_total + inundation.aggr_total
    #         self.results.append(inundation.df[['tide_elev', 'land_elev']])
    #         self.land_elev = inundation.result.y[2][-1]
    #         self.now = inundation.t_end + self.timestep
    #     elif status == -1:
    #         self.results.append(subset)
    #         self.degradation_total = self.degradation_total + (subset.land_elev.values[-1] - subset.land_elev.values[0])
    #         self.land_elev = subset.land_elev.values[-1]
    #         self.now = subset.index[-1] + self.timestep
    #     elif status == 1:
    #         self.results.append(subset)
    #         self.degradation_total = self.degradation_total + (subset.land_elev.values[-1] - subset.land_elev.values[0])
    #         self.results = pd.concat(self.results)
    #         self.land_elev = subset.land_elev.values[-1]
    #         self.now = subset.index[-1] + self.timestep
    #     elif status == -2:
    #         self.degradation_total = self.degradation_total + inundation.degr_total
    #         self.aggradation_total = self.aggradation_total + inundation.aggr_total
    #         self.results.append(inundation.df[['tide_elev', 'land_elev']])
    #         self.land_elev = inundation.result.y[2][-1]
    #         self.now = inundation.t_end + self.timestep

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
    tides: pd.Series
    elevation_start: float
    ssc_boundary: InitVar[float]
    bulk_density: InitVar[float]
    settling_rate: InitVar[float]
    constant_rates: InitVar[float]
    result: InundationResult = field(default=None, init=False)
    df: pd.DataFrame = field(default=None, init=False)
    aggradation_total: float = field(default=None, init=False)
    degradation_total: float = field(default=None, init=False)

    def __post_init__(self, ssc_boundary, bulk_density, settling_rate, constant_rates):
        self.timestep = self.tides.index[1] - self.tides.index[0]
        self.start = self.tides.index[0]
        self.end = self.tides.index[-1]
        self.period = self.end - self.start
        self.slack = self.tides.idxmax()
        self.slack_elevation = self.tides.max()
        tides_func = InterpolatedUnivariateSpline(x=datetime2num(self.tides.index), y=self.tides.values, k=3)
        self.params = Bunch(
            tides_func=tides_func,
            ssc_boundary=ssc_boundary,
            bulk_density=bulk_density,
            settling_rate=settling_rate,
            constant_rates=constant_rates,
        )

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

    def integrate(self, method='DOP853', dense_output=False):

        self.result = solve_ivp(
            fun=self.solve_odes,
            t_span=[datetime2num(self.start), datetime2num(self.end)],
            y0=[self.tides.values[0], 0.0, self.elevation_start, 0.0, 0.0, 0.0],
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
            logger.warning(f'{self.start} | Integration failed!\nparams={self.params}')
        if (self.df.aggradation < 0.0).any():
            logger.warning(f'{self.start} | Negative aggradation detected!\n{self.df.loc[self.df.aggradation < 0]}')
        if (self.df.aggradation_max < self.df.aggradation).any():
            logger.warning(
                f'{self.start} | Overextraction detected!\n{self.df.loc[self.df.aggradation_max < self.df.aggradation]}'
            )

    def _set_df(self):
        index = num2datetime(self.result.t)
        water_level = self.result.y[0]
        concentration = self.result.y[1]
        elevation = self.result.y[2]
        aggradation = self.result.y[3]
        aggradation_max = self.result.y[4]
        degradation = self.result.y[5]
        df = pd.DataFrame(
            data={
                'water_level': water_level,
                'elevation': elevation,
                'concentration': concentration,
                'aggradation': aggradation,
                'aggradation_max': aggradation_max,
                'degradation': degradation,
            },
            index=index,
        )
        if index[-1] != self.end:
            record = {}
            record["water_level"] = self.tides.at[self.end]
            record["elevation"] = df.elevation.values[-1] + calculate_rate_vector(
                t=self.end, rate=self.params.constant_rates, ref_t=df.index[-1]
            )
            record["concentration"] = 0.0
            record["aggradation"] = df.aggradation.values[-1]
            record["aggradation_max"] = df.aggradation_max.values[-1]
            record["degradation"] = df.degradation.values[-1]
            df.loc[self.end] = record

        self.degradation_total = df.degradation.values[-1]
        self.df = df

    def plot(self):

        fig, ax = plt.subplots(figsize=(15, 15), nrows=4, ncols=1, constrained_layout=True)
        fig.suptitle(f'Inundation at {self.start}', fontsize=16)

        aggr_max_mod_diff = self.df.aggradation_max - self.df.aggradation

        sns.lineplot(data=self.df.water_level, color='cornflowerblue', label='Tide', ax=ax[0])
        sns.lineplot(data=self.df.elevation, color='forestgreen', label='Land Surface', ax=ax[0])
        ax[0].set_ylabel(ylabel='Elevation (m)')

        sns.lineplot(data=self.df.concentration, color='saddlebrown', label="SSC", ax=ax[1])
        ax[1].set_ylabel(ylabel='Concentration (g/L)')

        sns.lineplot(data=self.df.aggradation_max, color='red', label='Max', ax=ax[2])
        sns.scatterplot(data=self.df.aggradation, label='Modeled', ax=ax[2])
        ax[2].set_ylabel(ylabel='Aggradation (m)')
        ax[2].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        sns.lineplot(data=aggr_max_mod_diff, color='black', linestyle=':', ax=ax[3])
        ax[3].set_ylabel(ylabel='Difference (m)\nMax - Modeled')
        ax[3].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax[3].fill_between(
            x=self.df.index, y1=aggr_max_mod_diff, where=aggr_max_mod_diff >= 0.0, color='forestgreen', alpha=0.3
        )
        ax[3].fill_between(x=self.df.index, y1=aggr_max_mod_diff, where=aggr_max_mod_diff < 0, color='red', alpha=0.3)
        for ax in ax:
            ax.axvline(self.slack, color='black', linestyle='--')
            ax.ticklabel_format(axis='y', useOffset=False)
