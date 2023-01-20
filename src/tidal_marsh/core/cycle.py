from __future__ import annotations

# from loguru import logger
from dataclasses import InitVar, dataclass, field

import pandas as pd
from scipy.interpolate import PchipInterpolator
from .inundation import Inundation

@dataclass
class Cycle:
    cycle_n: int
    water_levels: pd.Series
    start: pd.Timestamp = field(init=False)
    end: pd.Timestamp = field(init=False)
    period: pd.Timedelta = field(init=False)
    inundation: None | Inundation = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.start = self.water_levels.index[0]
        self.end = self.water_levels.index[-1]
        self.period = self.end - self.start

    def make_depth_func(self, elevation: float) -> PchipInterpolator:
        x = (self.water_levels.index - self.water_levels.index[0]).total_seconds()
        y = self.water_levels - elevation
        return PchipInterpolator(x=x, y=y, extrapolate=False)

    def find_inundation(self, elevation: float) -> None:
        depth = self.make_depth_func(elevation)
        start, mid, end = self.find_roots(depth)

        self.inundation = Inundation(
            depth=depth, start=start, mid=mid, end=end, time_ref=self.start, cycle_n=self.cycle_n
        )

    def find_roots(self, depth: PchipInterpolator) -> tuple[pd.Timedelta, ...]:
        roots = depth.roots()
        roots_d1dt = depth.derivative().roots()
        i = roots_d1dt[depth(roots_d1dt).argmax()]
        mid = pd.Timedelta(i, unit="s")
        if roots.size == 2:
            start = pd.Timedelta(roots[0] + 1, unit="s")
            end = pd.Timedelta(roots[1] - 1, unit="s")
        elif roots.size == 1:
            if roots[0] < i:
                start = pd.Timedelta(roots[0] + 1, unit="s")
                end = self.end - self.start
            else:
                start = pd.Timedelta(0)
                end = pd.Timedelta(roots[0] - 1, unit="s")
        elif roots.size == 0:
            start = pd.Timedelta(0)
            end = self.end - self.start

        return (start, mid, end)