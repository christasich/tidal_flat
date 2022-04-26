from dataclasses import InitVar, dataclass, field

import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from . import constants
from . import utils
from .inundation import Inundation


@dataclass
class Model:
    # input these fields
    water_levels: pd.Series
    initial_elevation: InitVar[float | int]
    # params
    ssc: InitVar[float | int | np.ndarray | pd.Series]
    grain_diameter: float
    grain_density: float
    bulk_density: float
    organic_rate: float = 0.0
    compaction_rate: float = 0.0
    subsidence_rate: float = 0.0
    solve_ivp_opts: dict = field(default_factory=dict)
    # diagnositic
    id: int = 0
    position: int = 0
    save_inundations: bool = field(repr=False, default=False)

    # auto initialize these fields
    aggradation_total: float = field(init=False, default=0.0)
    degradation_total: float = field(init=False, default=0.0)
    overextraction: float = field(init=False, default=0.0)
    inundations: list = field(init=False, default_factory=list)
    results: pd.DataFrame = field(init=False, default_factory=list)
    start: pd.Timestamp = field(init=False)
    end: pd.Timestamp = field(init=False)
    now: pd.Timestamp = field(init=False)
    elevation: float = field(init=False)
    timestep: pd.Timedelta = field(init=False)
    pbar_unit: pd.Timedelta = field(init=False)

    def __post_init__(
        self,
        initial_elevation: float | int,
        ssc: float | int | np.ndarray | pd.Series,
    ) -> None:
        self.start = self.now = self.water_levels.index[0]
        self.end = self.water_levels.index[-1]
        self.timestep = pd.Timedelta(self.water_levels.index.freq)
        self.logger = logger.patch(lambda record: record["extra"].update(model_time=self.now))
        self.update(index=self.start, water_level=self.water_levels.iat[0], elevation=initial_elevation)
        self.water_levels = self.water_levels.rename("water_level")
        # repeat ssc for each month if given a single number
        if isinstance(ssc, (float, int)):
            ssc = np.repeat(ssc, 12)
        elif isinstance(ssc, pd.Series):
            ssc = ssc.values

        # make a daily ssc index to match tides
        # set value at midpoint of each month
        # add one month on each side for better interpolation
        start = self.water_levels.index[0] - pd.DateOffset(months=1)
        end = self.water_levels.index[-1] + pd.DateOffset(months=1)
        ssc_index = pd.date_range(start=start, end=end, freq="MS") + pd.DateOffset(days=14)

        # make a series from ssc array and new index then upsample to 1D and interpolate.
        ssc_series = pd.Series(data=ssc[ssc_index.month - 1], index=ssc_index).asfreq("1D").interpolate()

        # remove trailing and lead months to match original index
        self.ssc = ssc_series.loc[self.water_levels.index[0]: self.water_levels.index[-1]]

        solve_ivp_kwargs = {
            'method': 'RK45',
            'dense_output': False,
            'first_step': None,
            'max_step': np.inf,
            'rtol': 1e-3,
            'atol': 1e-6,
        }
        solve_ivp_kwargs.update(self.solve_ivp_opts)
        self.solve_ivp_opts = solve_ivp_kwargs

    @property
    def period(self) -> pd.Timestamp:
        return self.end - self.start

    @property
    def constant_rates(self) -> float:
        return (self.organic_rate + self.compaction_rate + self.subsidence_rate) / constants.YEAR

    @property
    def settling_rate(self) -> float:
        return utils.stokes_settling(grain_diameter=self.grain_diameter, grain_density=self.grain_density)

    @property
    def inow(self) -> int:
        return self.water_levels.index.get_loc(self.now)

    @property
    def state(self) -> pd.Series:
        return pd.Series(data={"elevation": self.elevation}, index=self.now)

    def calculate_elevation(self, at: pd.Timestamp = None, to: pd.Timestamp = None) -> float | pd.Series:
        if at:
            elapsed_seconds = (at - self.now).total_seconds()
            return self.constant_rates * elapsed_seconds + self.elevation
        elif to:
            index = self.water_levels.loc[self.now: to].index
            elapsed_seconds = (index - self.now).total_seconds()
            return (self.constant_rates * elapsed_seconds).values + self.elevation

    # def validate_inundation(self, inundation: Inundation) -> None:
    #     for result in [inundation.flood, inundation.ebb]:
    #         if result.success is False:
    #             self.logger.warning(
    #                 f"Integration failed using {inundation.solve_ivp_opts} with message: {result.message}")
    #     if (inundation.data.aggradation < 0.0).any():
    #         self.logger.warning("Solution contains negative aggradations.")
    #         save = True
    #     if inundation.data.aggradation[-1] > inundation.slack_depth:
    #         self.logger.warning("More sediment extracted than possible.")
    #         save = True
    #     if inundation.data.depth[-1] > 0:
    #         self.logger.warning("Inundation ends above platform.")
    #         save = True
    #     if save:
    #         self.invalid_inundations.append(inundation)

    def update(self, index: pd.Timestamp, water_level: float, elevation: float, skip='0S') -> None:
        self.logger.trace(f"Updating results: Date={index}, Water Level={water_level}, Elevation={elevation}")
        self.results.append({"index": index, "water_level": water_level, "elevation": elevation})
        self.now = index + pd.Timedelta(skip)
        self.elevation = elevation

    def find_next_inundation(self) -> pd.DataFrame | None:

        n = pd.Timedelta("1D")

        subset = self.water_levels.loc[self.now: self.now + n].to_frame(name="water_level")
        subset["elevation"] = self.calculate_elevation(to=subset.index[-1])
        roots = utils.find_roots(a=subset.water_level.values, b=subset.elevation.values)

        while len(roots) < 2:
            if subset.index[-1] == self.end and len(roots) == 1:
                self.logger.trace(f"Partial inundation remaining.")
                return subset.iloc[roots[0] + 1:]
            elif subset.index[-1] == self.end and len(roots) == 0:
                self.logger.trace(f"No inundations remaining.")
                return None
            else:
                n = n * 1.5
                self.logger.trace(f"Expanding search window to {n}.")
                subset = self.water_levels.loc[self.now: self.now + n].to_frame(name="water_level")
                subset["elevation"] = self.calculate_elevation(to=subset.index[-1])
                roots = utils.find_roots(a=subset.water_level.values, b=subset.elevation.values)
        self.logger.trace(f"Found complete inundation.")
        return subset.iloc[roots[0] + 1: roots[1] + 1]

    def step(self) -> None:
        subset = self.find_next_inundation()
        if subset is not None:
            self.logger.trace(f"Initializing Inundation at {subset.index[0]}.")
            inundation = Inundation(
                water_levels=subset.water_level,
                initial_elevation=subset.elevation.iat[0],
                ssc_boundary=self.ssc[self.now.month - 1],
                bulk_density=self.bulk_density,
                settling_rate=self.settling_rate,
                constant_rates=self.constant_rates,
                solve_ivp_opts=self.solve_ivp_opts,
            )
            if self.save_inundations:
                self.inundations.append(inundation)
            inundation.integrate()
            # if inundation.valid is False:
            #     self.logger.warning(f'Invalid inundation at index {len(self.inundations)}.')

            for record in inundation.data[['water_level', 'elevation']].reset_index().to_dict(orient='records'):
                self.update(index=record['index'], water_level=record['water_level'], elevation=record['elevation'], skip='1H')
        else:
            self.logger.trace("No inunundations remaining.")
            elevation = self.calculate_elevation(at=self.end)
            self.update(index=self.end, water_level=np.nan, elevation=elevation)

    def initialize(self, end_date, period):
        if end_date:
            self.end = pd.to_datetime(end_date)
        if period:
            self.end = self.start + pd.to_timedelta(period)
        if self.period < pd.Timedelta('365.25D'):
            self.pbar_unit = pd.Timedelta('1D')
            unit = "Day"
        else:
            self.pbar_unit = pd.Timedelta('365.25D')
            unit = "Year"
        self.pbar = tqdm(
            desc=f"W{self.position:02}:{self.id:04}",
            total=round(self.period / self.pbar_unit, 2),
            leave=True,
            unit=unit,
            dynamic_ncols=True,
            position=self.position,
            smoothing=0,
            postfix={"Date": self.now.strftime("%Y-%m-%d"), "Elevation": self.elevation},
        )
        if self.water_levels.iat[0] > self.elevation:
            self.logger.debug(f"Tide starts above platform. Skipping first inundation.")
            elevation = self.calculate_elevation(to=self.end)
            i = (elevation > self.water_levels).argmax()
            self.update(index=self.water_levels.index[i],
                        water_level=self.water_levels.index[i], elevation=elevation[i])

    def uninitialize(self) -> None:
        self.results = pd.DataFrame.from_records(data=self.results, index="index").squeeze()
        self.pbar.close()
        self.logger.info('Simulation completed. Exiting.')

    def run(self, end_date=None, period=None) -> None:
        self.initialize(end_date, period)
        while self.now < self.end:
            self.step()
            new_n = round((self.now - self.start) / self.pbar_unit, 2)
            self.pbar.set_postfix({"Date": self.now.strftime("%Y-%m-%d"), "Elevation": self.elevation}, refresh=False)
            self.pbar.update(n=new_n - self.pbar.n)
        self.uninitialize()

    # def run(self, stop=None, steps=None) -> None:
    #     if stop is None:
    #         stop = self.end
    #     if steps is None:
    #         steps = float("inf")
    #     stop = pd.to_datetime(stop)
    #     self.initialize()
    #     while (self.now < stop) and (len(self.inundations) < steps):
    #         self.step()
    #         new_n = (self.now - self.start).ceil("D").days
    #         self.pbar.set_postfix({"Date": self.now.strftime("%Y-%m-%d"), "Elevation": self.elevation}, refresh=False)
    #         self.pbar.update(n=new_n - self.pbar.n)
    #     self.uninitialize()

    def plot_results(self, freq="10T") -> None:

        data = pd.concat([self.water_levels, self.results], axis=1).resample(freq).mean()

        _, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)

        sns.lineplot(data=data.water_level, color="cornflowerblue", alpha=0.6, ax=ax)
        sns.lineplot(data=data.elevation, color="forestgreen", ax=ax)

        plt.xlim((data.index[0], data.index[-1]))
        plt.ylim((data.elevation.min(), data.water_level.max()))
