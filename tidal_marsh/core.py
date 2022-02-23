from dataclasses import InitVar, dataclass, field

import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from . import constants, utils
from .inundation import Inundation


@dataclass
class Model:
    # input these fields
    water_levels: pd.Series
    initial_elevation: InitVar[float | int]
    # params
    ssc: InitVar[float | int | np.ndarray]
    grain_diameter: float
    grain_density: float
    bulk_density: float
    organic_rate: float = 0.0
    compaction_rate: float = 0.0
    subsidence_rate: float = 0.0
    # diagnositic
    id: int = 0
    debug: bool = field(repr=False, default=False)

    # auto initialize these fields
    aggradation_total: float = field(init=False, default=0.0)
    degradation_total: float = field(init=False, default=0.0)
    overextraction: float = field(init=False, default=0.0)
    inundations: list = field(init=False, default_factory=list)
    results: list = field(init=False, default_factory=list)
    start: pd.Timestamp = field(init=False)
    end: pd.Timestamp = field(init=False)
    now: pd.Timestamp = field(init=False)
    elevation: float = field(init=False)
    period: pd.Timedelta = field(init=False)
    timestep: pd.Timedelta = field(init=False)

    def __post_init__(
        self,
        initial_elevation: float | int,
        ssc: float | int | np.ndarray,
    ) -> None:
        self.water_levels = self.water_levels.rename("water_level")
        # repeat ssc for each month if given a single number
        if isinstance(ssc, (float, int)):
            ssc = np.repeat(ssc, 12)

        # make a daily ssc index to match tides
        # set value at midpoint of each month
        # add one month on each side for better interpolation
        start = self.water_levels.index[0] - pd.DateOffset(months=1)
        end = self.water_levels.index[-1] + pd.DateOffset(months=1)
        ssc_index = pd.date_range(start=start, end=end, freq="MS") + pd.DateOffset(days=14)

        # make a series from ssc array and new index then upsample to 1D and interpolate.
        ssc_series = pd.Series(data=ssc[ssc_index.month - 1], index=ssc_index).asfreq("1D").interpolate()

        # remove trailing and lead months to match original index
        self.ssc = ssc_series.loc[self.water_levels.index[0] : self.water_levels.index[-1]]

        self.start = self.water_levels.index[0]
        self.end = self.water_levels.index[-1]
        self.period = self.water_levels.index[-1] - self.water_levels.index[0]
        self.timestep = pd.Timedelta(self.water_levels.index.freq)
        self.update(index=self.start, elevation=initial_elevation)
        self.pbar = tqdm(
            desc=f"{self.id:04}",
            total=self.period.ceil("D").days,
            leave=True,
            unit="Day",
            position=self.id + 1,
            postfix={"Date": self.now.strftime("%Y-%m-%d"), "Elevation": self.elevation},
        )

        if self.water_levels.iat[0] > self.elevation:
            logger.info("Tide starts above platform. Skipping first inundation.")
            elevation = self.calculate_elevation(to=self.end)
            i = (elevation > self.water_levels).argmax()
            self.update(index=self.water_levels.index[i], elevation=elevation[i])

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
    def state(self):
        return pd.Series(data={"elevation": self.elevation}, index=self.now)

    def calculate_elevation(self, at=None, to=None):
        if at:
            elapsed_seconds = (at - self.now).total_seconds()
            return self.constant_rates * elapsed_seconds + self.elevation
        elif to:
            index = self.water_levels.loc[self.now : to].index
            elapsed_seconds = (index - self.now).total_seconds()
            return (self.constant_rates * elapsed_seconds).values + self.elevation

    def update(self, index, elevation):
        logger.debug(f"Updating results - Date: {index}, Elevation={elevation}")
        self.results.append({"index": index, "elevation": elevation})
        self.now = index
        self.elevation = elevation

    def find_next_inundation(self):

        n = pd.Timedelta("1D")

        subset = self.water_levels.loc[self.now : self.now + n].to_frame(name="water_level")
        subset["elevation"] = self.calculate_elevation(to=subset.index[-1])
        roots = utils.find_roots(a=subset.water_level.values, b=subset.elevation.values)

        while len(roots) < 2:
            if subset.index[-1] == self.end and len(roots) == 1:
                return subset.iloc[roots[0] + 1 :]
            elif subset.index[-1] == self.end and len(roots) == 0:
                return
            else:
                n = n * 1.5
                subset = self.water_levels.loc[self.now : self.now + n].to_frame(name="water_level")
                subset["elevation"] = self.calculate_elevation(to=subset.index[-1])
                roots = utils.find_roots(a=subset.water_level.values, b=subset.elevation.values)
        return subset.iloc[roots[0] + 1 : roots[1] + 2]

    def step(self):
        subset = self.find_next_inundation()
        if subset is not None:

            inundation = Inundation(
                water_levels=subset.water_level,
                initial_elevation=subset.elevation.iat[0],
                ssc_boundary=self.ssc[self.now.month - 1],
                bulk_density=self.bulk_density,
                settling_rate=self.settling_rate,
                constant_rates=self.constant_rates,
            )
            logger.debug(f"Processing inundation - Date: {inundation.start}, Elevation={inundation.initial_elevation}")
            self.update(index=inundation.start, elevation=inundation.initial_elevation)
            if self.debug is True:
                self.inundations.append(inundation)
            try:
                inundation.integrate()
            except:
                logger.exception("Problem with integration!")
            else:
                self.overextraction += inundation.overextraction
                self.update(index=inundation.data.index[-1], elevation=inundation.data.elevation.iat[-1])
        else:
            elevation = self.calculate_elevation(at=self.end)
            self.update(index=self.end, elevation=elevation)

    def run(self, stop=None, steps=None):
        if stop is None:
            stop = self.end
        if steps is None:
            steps = float("inf")
        stop = pd.to_datetime(stop)
        while (self.now < stop) and (len(self.inundations) < steps):
            self.step()
            new_n = (self.now - self.start).ceil("D").days
            # self.pbar.n = (self.now - self.start).ceil("D").days
            self.pbar.set_postfix({"Date": self.now.strftime("%Y-%m-%d"), "Elevation": self.elevation}, refresh=False)
            self.pbar.update(n=new_n - self.pbar.n)
        if self.now == self.end:
            logger.info("No more inundations to process. Exiting.")
            self.results = pd.DataFrame.from_records(data=self.results, index="index").squeeze()
            self.pbar.close()

    def plot_results(self, freq="10T"):

        data = pd.concat([self.water_levels, self.results], axis=1).resample(freq).mean()

        fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)

        sns.lineplot(data=data.water_level, color="cornflowerblue", alpha=0.6, ax=ax)
        sns.lineplot(data=data.elevation, color="forestgreen", ax=ax)

        plt.xlim((data.index[0], data.index[-1]))
        plt.ylim((data.elevation.min(), data.water_level.max()))
