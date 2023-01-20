import contextlib
from dataclasses import InitVar, dataclass, field
import pickle
from typing import Any

import pandas as pd
from loguru import logger
from sklearn.utils import Bunch
from tqdm.auto import tqdm

from .inundation import Inundation
from .platform import Platform
from .cycle import Cycle
from ..constants import CYCLES_PER_DAY, CYCLES_PER_YEAR, YEAR, GRAVITY, WATER_DENSITY, WATER_VISCOSITY
from ..tides import Tides


@dataclass
class Model:
    # input these fields
    tides: Tides
    platform: Platform

    ssc: float
    grain_diameter: float
    bulk_density: float
    grain_density: float
    org_sed: float = 0.0
    compaction: float = 0.0
    deep_sub: float = 0.0
    settling_rate: float = field(init=False)

    solve_ivp_opts: InitVar[dict] = None
    pbar_opts: InitVar[dict] = None
    results: Bunch = field(init=False, default_factory=Bunch)
    start: pd.Timestamp = field(init=False)
    end: pd.Timestamp = field(init=False)
    period: pd.Timedelta = field(init=False)
    now: pd.Timestamp = field(init=False)
    pos: int = field(init=False, default=0)
    inundations: list = field(init=False, default_factory=list)
    remaining_cycles: pd.DataFrame = field(init=False)

    @property
    def tz(self) -> Any:
        return self.tides.data.index.tz

    @property
    def freq(self) -> Any:
        return self.tides.levels.index.freq

    def __post_init__(self, solve_ivp_opts, pbar_opts) -> None:
        if self.bkgrd_offset() > 0:
            self.tides = self.tides.raise_sea_level(-self.bkgrd_offset())
        self.start = self.now = self.tides.start
        self.end = self.tides.end
        self.period = self.tides.period
        self.logger = logger.patch(lambda record: record["extra"].update(model_time=self.now))

        self.settling_rate = stokes_settling(grain_diameter=self.grain_diameter, grain_density=self.grain_density)
        self.sed_params = {"ssc": self.ssc, "bulk_density": self.bulk_density, "settling_rate": self.settling_rate}

        self.remaining_cycles = self.tides.cycles.loc[self.pos :]

        if pbar_opts:
            self.setup_pbar(**pbar_opts)
        else:
            self.setup_pbar()

    @property
    def current_elevation(self) -> float:
        return self.platform.elevation_ref + self.platform.aggradation + self.bkgrd_offset(self.platform.elapsed)

    @property
    def change_per_year(self) -> float:
        net = self.platform.aggradation + self.bkgrd_offset(self.now - self.start)
        years = (self.now - self.start) / YEAR
        return net / years

    def pickle(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def setup_pbar(self, **kwargs: Any) -> None:
        user_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        cycles_per_unit = CYCLES_PER_YEAR
        unit = "year"
        if len(self.remaining_cycles) < cycles_per_unit:
            unit = "day"
            cycles_per_unit = CYCLES_PER_DAY
        total = round(self.remaining_cycles.shape[0] / cycles_per_unit, 2)
        self.pbar_opts = {
            "total": total,
            "unit": unit,
            "dynamic_ncols": True,
            "maxinterval": 1,
            "smoothing": 0,
            "postfix": {"Elevation": f"{self.platform.elevation_ref:.3f} ({0.0:+.3f}) m"},
        } | user_kwargs

        self.pbar = tqdm(**self.pbar_opts)
        self.pbar.cycles_per_unit = cycles_per_unit

    def update_pbar(self) -> None:
        net = self.platform.aggradation + self.bkgrd_offset(self.now - self.start)
        years = (self.now - self.start) / YEAR
        net_per_year = net / years if years > 0 else 0
        self.pbar.set_postfix(
            {"Elevation": f"{self.platform.elevation_ref + net:.3f} m | {net_per_year*100:+.2f} cm/yr"}, refresh=False
        )
        step = round((self.pos + 1) / self.pbar.cycles_per_unit, 2) - self.pbar.n
        self.pbar.update(n=step)

    def bkgrd_offset(self, time: pd.Timedelta = YEAR) -> float:
        return (self.org_sed + self.compaction + self.deep_sub) * time / YEAR

    def next_cycle(self) -> Cycle | None:
        check = self.remaining_cycles.high > self.platform.surface
        return self.tides.make_cycle(n=check.idxmax()) if check.any() else None

    def step(self) -> None:
        cycle = self.next_cycle()
        if cycle is not None:
            cycle.find_inundation(elevation=self.platform.surface)
            self.inundate_platform(cycle)
            self.update(now=cycle.end, pos=cycle.cycle_n)
        else:
            self.update(now=self.end, pos=self.tides.cycles.index[-1])

    def update(self, now: pd.Timestamp, pos: int) -> None:
        self.now = now
        self.pos = pos
        self.remaining_cycles = self.tides.cycles.loc[self.pos + 1 :]
        self.update_pbar()

    def inundate_platform(self, cycle: Cycle) -> None:
        elapsed = cycle.end - self.now
        cycle.inundation.solve(**self.sed_params)
        self.platform.update(time=elapsed, aggradation=cycle.inundation.aggradation)
        self.record_inundation(cycle.inundation)

    def record_inundation(self, inundation: Inundation):
        summary = inundation.summary
        i = 0
        try:
            if summary.start == self.inundations[-1].end:
                i = self.inundations[-1].i
            else:
                i = self.inundations[-1].i + 1
        except IndexError:
            pass
        finally:
            summary["i"] = i
            self.inundations.append(summary)

    def run(self) -> None:
        with contextlib.suppress(KeyboardInterrupt):
            while self.now < self.end:
                self.step()

    def close(self, compile_results=True) -> None:
        self.pbar.close()
        if compile_results:
            self.set_results()
        self.logger.info("Simulation completed. Exiting.")

    def set_results(self) -> Bunch:
        self.results.totals = self.process_totals()
        self.results.inundations = self.process_inundations()

    def process_totals(self):
        index = pd.date_range(self.start, self.end, freq=self.freq)
        subsidence = pd.Series(data=self.bkgrd_offset(index - self.start), index=index, name="subsidence")
        aggradation = self.platform.report()
        totals = pd.concat([aggradation, subsidence], axis=1).ffill()
        totals["net"] = totals.aggradation + totals.subsidence
        totals["elevation"] = self.platform.elevation_ref + totals.net
        return totals

    def process_inundations(self):
        inundations = pd.concat(self.inundations, axis=1).T.sort_values("i")
        return inundations.groupby("i").apply(process_inundation)
        # inundations = inundations.groupby("i").apply(process_inundation)
        # return (
        #     inundations.resample(freq, on="start")
        #     .agg(
        #         hydroperiod=("hydroperiod", "mean"),
        #         aggradation=("aggradation", "mean"),
        #         max_depth=("max_depth", "mean"),
        #         total=("num_cycles", "sum"),
        #     )
        #     .rename_axis(index="datetime")
        # )

    # def process_tides(self, freq="A"):
    #     return self.tides.summarize(freq=freq)


def process_inundation(df):
    hp_weights = df.hydroperiod / df.hydroperiod.sum()
    s = df.reset_index().agg(
        {
            "start": "min",
            "end": "max",
            "hydroperiod": "sum",
            "aggradation": "mean",
            "max_depth": "max",
            "valid": "all",
        }
    )
    s["mean_depth"] = (df.mean_depth * hp_weights).sum()
    s["cycles"] = df.index.values
    s["num_cycles"] = df.index.shape[0]
    return s


def stokes_settling(
    grain_diameter: float,
    grain_density: float,
    fluid_density: float = WATER_DENSITY,
    fluid_viscosity: float = WATER_VISCOSITY,
    gravity: float = GRAVITY,
) -> float:
    return (2 / 9 * (grain_density - fluid_density) / fluid_viscosity) * gravity * (grain_diameter / 2) ** 2
