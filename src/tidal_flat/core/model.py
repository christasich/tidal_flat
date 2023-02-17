import contextlib
import pickle
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from loguru import logger
from sklearn.utils import Bunch
from tqdm.auto import tqdm

from tidal_flat.constants import CYCLES_PER_DAY, CYCLES_PER_YEAR, GRAVITY, WATER_DENSITY, WATER_VISCOSITY, YEAR

from .inundation import Inundation
from .platform import Platform
from .tides import Cycle, Tides


@dataclass
class Model:
    tides: Tides
    platform: Platform

    ssc: float
    grain_diameter: float
    bulk_density: float
    grain_density: float = 2.65e3
    org_sed: float = 0.0
    compaction: float = 0.0
    deep_sub: float = 0.0
    settling_rate: float = field(init=False)
    start: pd.Timestamp = field(init=False)
    end: pd.Timestamp = field(init=False)
    period: pd.Timedelta = field(init=False)
    now: pd.Timestamp = field(init=False)
    pos: int = field(init=False, default=0)
    inundations: list = field(init=False, default_factory=list)
    remaining_cycles: pd.DataFrame = field(init=False)
    pbar_unit: float = field(init=False, default=CYCLES_PER_YEAR)

    @property
    def tz(self) -> Any:
        return self.tides.data.index.tz

    @property
    def freq(self) -> Any:
        return self.tides.levels.index.freq

    def __post_init__(self) -> None:
        if self.bkgrd_offset() > 0:
            self.tides = self.tides.raise_sea_level(-self.bkgrd_offset())
        self.start = self.now = self.tides.start
        self.end = self.tides.end
        self.period = self.tides.period
        self.logger: logger = logger.patch(lambda record: record['extra'].update(model_time=self.now))

        self.settling_rate = self.stokes_settling(grain_diameter=self.grain_diameter, grain_density=self.grain_density)
        self.sed_params = {'ssc': self.ssc, 'bulk_density': self.bulk_density, 'settling_rate': self.settling_rate}

        self.remaining_cycles = self.tides.cycles.loc[self.pos :]
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
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def setup_pbar(self) -> None:
        unit = 'year'
        if len(self.remaining_cycles) < self.pbar_unit:
            unit = 'day'
            self.pbar_unit = CYCLES_PER_DAY
        total = round(self.remaining_cycles.shape[0] / self.pbar_unit, 2)
        self.pbar_opts = {
            'total': total,
            'unit': unit,
            'dynamic_ncols': True,
            'maxinterval': 1,
            'smoothing': 0,
            'postfix': {'Elevation': f'{self.platform.elevation_ref:.3f} ({0.0:+.3f}) m'},
        }
        self.pbar = tqdm(**self.pbar_opts)

    def update_pbar(self) -> None:
        net = self.platform.aggradation + self.bkgrd_offset(self.now - self.start)
        years = (self.now - self.start) / YEAR
        net_per_year = net / years if years > 0 else 0
        self.pbar.set_postfix(
            {'Elevation': f'{self.platform.elevation_ref + net:.3f} m | {net_per_year*100:+.2f} cm/yr'}, refresh=False
        )
        step = round((self.pos + 1) / self.pbar_unit, 2) - self.pbar.n
        self.pbar.update(n=step)

    def bkgrd_offset(self, time: pd.Timedelta | pd.TimedeltaIndex = YEAR) -> Any:
        return (self.org_sed + self.compaction + self.deep_sub) * time / YEAR

    def next_cycle(self) -> Cycle | None:
        check = self.remaining_cycles.high > self.platform.surface
        return self.tides.make_cycle(n=int(check.idxmax())) if check.any() else None

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
            summary['i'] = i
            self.inundations.append(summary)

    def run(self) -> None:
        with contextlib.suppress(KeyboardInterrupt):
            while self.now < self.end:
                self.step()
            self.pbar.close()
            self.logger.info('Simulation completed. Exiting.')

    def summarize(self) -> Bunch:
        return Bunch(platform=self.elevation_history(), inundations=self.inundation_history())

    def elevation_history(self):
        index = pd.date_range(self.start, self.end, freq=self.freq)
        subsidence = pd.Series(data=self.bkgrd_offset(index - self.start), index=index, name='subsidence')
        aggradation = self.platform.report()
        totals = pd.concat([aggradation, subsidence], axis=1).ffill()
        totals['net'] = totals.aggradation + totals.subsidence
        totals['elevation'] = self.platform.elevation_ref + totals.net
        return totals

    def inundation_history(self):
        inundations = pd.concat(self.inundations, axis=1).T.sort_values('i')
        return inundations.groupby('i').apply(self.process_inundations)

    @staticmethod
    def process_inundations(df):
        hp_weights = df.hydroperiod / df.hydroperiod.sum()
        s = df.reset_index().agg(
            {
                'start': 'min',
                'end': 'max',
                'hydroperiod': 'sum',
                'aggradation': 'mean',
                'max_depth': 'max',
                'valid': 'all',
            }
        )
        s['mean_depth'] = (df.mean_depth * hp_weights).sum()
        s['cycles'] = df.index.values
        s['num_cycles'] = df.index.shape[0]
        return s

    @staticmethod
    def stokes_settling(
        grain_diameter: float,
        grain_density: float,
        fluid_density: float = WATER_DENSITY,
        fluid_viscosity: float = WATER_VISCOSITY,
        gravity: float = GRAVITY,
    ) -> float:
        return (2 / 9 * (grain_density - fluid_density) / fluid_viscosity) * gravity * (grain_diameter / 2) ** 2
