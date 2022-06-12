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
    ssc: float
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
    pbar_name: str = None
    save_inundations: bool = field(repr=False, default=False)

    # auto initialize these fields
    total_aggradation: float = field(init=False, default=0.0)
    total_subsidence: float = field(init=False, default=0.0)
    inundations: list = field(init=False, default_factory=list)
    invalid_inundations: list = field(init=False, default_factory=list)
    results: pd.DataFrame = field(init=False, default_factory=list)
    start: pd.Timestamp = field(init=False)
    end: pd.Timestamp = field(init=False)
    now: pd.Timestamp = field(init=False)
    elevation: float = field(init=False)
    timestep: pd.Timedelta = field(init=False)
    pbar_unit: pd.Timedelta = field(init=False)

    def __post_init__(self, initial_elevation: float | int) -> None:
        self.start = self.now = self.water_levels.index[0]
        self.end = self.water_levels.index[-1]
        self.timestep = pd.Timedelta(self.water_levels.index.freq)
        self.logger = logger.patch(lambda record: record["extra"].update(model_time=self.now))
        self.update(index=self.start, water_level=self.water_levels.iat[0], elevation=initial_elevation)
        self.water_levels = self.water_levels.rename("water_level")
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

    def update(self, index: pd.Timestamp, water_level: float, elevation: float) -> None:
        self.logger.trace(f"Updating results: Date={index}, Water Level={water_level}, Elevation={elevation}")
        self.results.append({"index": index, "water_level": water_level, "elevation": elevation})
        self.now = index
        self.elevation = elevation

    def find_next_inundation(self) -> pd.DataFrame | None:

        valid = False
        while valid is False:

            n = pd.Timedelta("1D")

            subset = self.water_levels.loc[self.now: self.now + n].to_frame(name="water_level")
            subset["elevation"] = self.calculate_elevation(to=subset.index[-1])
            roots = utils.find_roots(a=subset.water_level.values, b=subset.elevation.values + 1e-3)

            while len(roots) < 2:
                if subset.index[-1] == self.end and len(roots) == 1:
                    self.logger.trace(f"Partial inundation remaining.")
                    roots = np.append(roots, None)
                elif subset.index[-1] == self.end and len(roots) == 0:
                    self.logger.trace(f"No inundations remaining.")
                    return None
                else:
                    n = n * 1.5
                    self.logger.trace(f"Expanding search window to {n}.")
                    subset = self.water_levels.loc[self.now: self.now + n].to_frame(name="water_level")
                    subset["elevation"] = self.calculate_elevation(to=subset.index[-1])
                    roots = utils.find_roots(a=subset.water_level.values, b=subset.elevation.values + 1e-3)
            if roots[1]:
                i = subset.iloc[roots[0] + 1: roots[1] + 1]
            elif roots[1] is None:
                i = subset.iloc[roots[0] + 1:]
            if i.shape[0] <= 3:
                self.logger.debug(f'Inundation is too small (len={i.shape[0]}). Skipping.')
                self.now = i.index[-1] + pd.Timedelta('1H')
                continue
            else:
                valid = True
        self.logger.trace(f"Initializing Inundation at {subset.index[0]}.")
        inundation = Inundation(
            water_levels=i.water_level,
            initial_elevation=i.elevation.iat[0],
            ssc_boundary=self.ssc,
            bulk_density=self.bulk_density,
            settling_rate=self.settling_rate,
            constant_rates=self.constant_rates,
            solve_ivp_opts=self.solve_ivp_opts,
        )
        return inundation

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
            desc=self.pbar_name,
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
            self.total_subsidence = self.total_subsidence + (elevation[i] - self.elevation)
            self.update(index=self.water_levels.index[i],
                        water_level=self.water_levels[i], elevation=elevation[i])

    def uninitialize(self) -> None:
        self.results = pd.DataFrame.from_records(data=self.results, index="index").squeeze()
        self.pbar.close()
        self.logger.info('Simulation completed. Exiting.')

    def step(self) -> None:
        inundation = self.find_next_inundation()
        if inundation is not None:
            if self.save_inundations:
                self.inundations.append(inundation)
            inundation.integrate()
            if not inundation.valid:
                self.invalid_inundations.append(inundation)

            self.total_subsidence = self.total_subsidence + (inundation.initial_elevation - self.elevation)
            for record in inundation.result[['water_level', 'elevation']].reset_index().to_dict(orient='records'):
                self.update(index=record['index'], water_level=record['water_level'],
                            elevation=record['elevation'])
            self.total_aggradation = self.total_aggradation + inundation.total_aggradation
            self.total_subsidence = self.total_subsidence + inundation.total_subsidence

            if inundation.end == self.end:
                self.logger.trace("No inunundations remaining.")
            else:
                i = inundation.end + pd.Timedelta('1H')
                elevation = self.calculate_elevation(at=i)
                self.total_subsidence = self.total_subsidence + (elevation - self.elevation)
                self.update(index=i, water_level=self.water_levels[i], elevation=elevation)
        else:
            self.logger.trace("No inunundations remaining.")
            elevation = self.calculate_elevation(at=self.end)
            self.total_subsidence = self.total_subsidence - (self.elevation - elevation)
            self.update(index=self.end, water_level=np.nan, elevation=elevation)

    def run(self, end_date=None, period=None) -> None:
        self.initialize(end_date, period)
        while self.now < self.end:
            self.step()
            new_n = round((self.now - self.start) / self.pbar_unit, 2)
            self.pbar.set_postfix({"Date": self.now.strftime("%Y-%m-%d"), "Elevation": self.elevation}, refresh=False)
            self.pbar.update(n=new_n - self.pbar.n)
        self.uninitialize()

    def plot_results(self, freq="10T") -> None:

        # data = pd.concat([self.water_levels, self.results], axis=1).resample(freq).mean()
        data = self.results.resample(freq).mean()

        _, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)

        sns.lineplot(data=data.water_level, color="cornflowerblue", alpha=0.6, ax=ax)
        sns.lineplot(data=data.elevation, color="forestgreen", ax=ax)

        plt.xlim((data.index[0], data.index[-1]))
        plt.ylim((data.elevation.min(), data.water_level.max()))
