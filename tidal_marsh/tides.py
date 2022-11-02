from __future__ import annotations
from dataclasses import InitVar, dataclass, field
from copy import deepcopy

import numpy as np
import pandas as pd
from loguru import logger
import importlib.resources as pkg_resources

from .utils import find_pv
from .inundation import Inundation

from scipy.interpolate import PchipInterpolator


# @dataclass
# class Cycle:
#     cycle_id: int
#     water_levels: pd.Series
#     start: pd.Timestamp = field(init=False)
#     end: pd.Timestamp = field(init=False)
#     slack: pd.Series = field(init=False)
#     period: pd.Timedelta = field(init=False)
#     inundation: Inundation = None

#     def __post_init__(self) -> None:
#         self.start = self.water_levels.index[0]
#         self.end = self.water_levels.index[-1]
#         self.period = self.end - self.start
#         self.lows = self.water_levels.iloc[[0, -1]]
#         self.high = self.water_levels.iloc[[self.water_levels.argmax()]].reset_index().squeeze().rename("high")

# def find_inundation(self, elevation: float) -> Inundation:
#     depth = PchipInterpolator(
#         x=self.water_levels.index.astype(int) / 10**9,
#         y=self.water_levels.values - elevation,
#         extrapolate=False,
#     )
#     roots = depth.roots()
#     if roots.size == 2:
#         r1 = roots[0] + 1
#         r2 = roots[1] - 1
#     elif roots.size == 1:
#         if roots[0] < self.high.datetime.timestamp():
#             r1 = roots[0] + 1
#             r2 = self.water_levels.index[-1].timestamp()
#         else:
#             r1 = self.water_levels.index[0].timestamp()
#             r2 = roots[0] - 1
#     elif roots.size == 0 and (self.water_levels > elevation).all():
#         r1 = self.water_levels.index[0].timestamp()
#         r2 = self.water_levels.index[-1].timestamp()
#     start = pd.to_datetime(r1, unit="s", origin="unix", utc=True).tz_convert(self.water_levels.index.tz)
#     end = pd.to_datetime(r2, unit="s", origin="unix", utc=True).tz_convert(self.water_levels.index.tz)

#     return Inundation(depth=depth, start=start, end=end)


@dataclass
class Tides:
    water_levels: pd.Series
    highs: pd.Series = field(init=False)
    lows: pd.Series = field(init=False)
    start: pd.Timestamp = field(init=False)
    end: pd.Timestamp = field(init=False)
    period: pd.Timedelta = field(init=False)
    cycles: pd.DataFrame = field(init=False)
    summary: pd.DataFrame = field(init=False)
    slr: float = 0.0
    amp_factor: float = 0.0

    def __post_init__(self) -> None:
        self.water_levels = self.water_levels.rename("water_levels").rename_axis(index="datetime")
        highs, lows = find_pv(data=self.water_levels, distance="8H")
        self.low_locs = lows.index
        self.high_locs = highs.loc[lows.index[0] : lows.index[-1]].index
        self.water_levels = self.water_levels.loc[lows.index[0] : lows.index[-1]]

        self.start = self.water_levels.index[0]
        self.end = self.water_levels.index[-1]
        self.period = self.end - self.start
        self.update()

    def raise_sea_level(self, slr: float) -> Tides:
        copy = deepcopy(self)
        copy.slr = slr
        years = (self.water_levels.index - self.start) / pd.to_timedelta("365.2425D")
        copy.water_levels += slr * years
        copy.update()
        return copy

    def amplify(self, factor: float) -> None:
        copy = deepcopy(self)
        copy.amp_factor = factor
        copy.water_levels = (copy.water_levels - copy.water_levels.mean()) * factor + copy.water_levels.mean()
        copy.update()
        return copy

    def update(self):
        self.highs = self.water_levels.loc[self.high_locs]
        self.lows = self.water_levels.loc[self.low_locs]
        self.cycles = pd.DataFrame(
            data={
                "start": self.low_locs[:-1],
                "slack": self.high_locs,
                "end": self.low_locs[1:],
                "period": self.low_locs[1:] - self.low_locs[:-1],
                "low1": self.lows.iloc[:-1].values,
                "high": self.highs.values,
                "low2": self.lows.iloc[1:].values,
            },
        ).rename_axis("cycle")
        self.cycles["range"] = self.cycles.high - (self.cycles.low1 + self.cycles.low2) / 2
        self.summary = summarize_tides(data=self.water_levels)

        # def make_cycle(self, n: int) -> pd.Series:
        #     cycle = self.cycles.loc[n]
        #     return Cycle(cycle_id=n, water_levels=self.water_levels.loc[cycle.start : cycle.end])

    def summarize(self, freq="A", start=None, end=None) -> pd.DataFrame:
        if start is None:
            start = self.start
        if end is None:
            end = self.end
        return self.water_levels.loc[start:end].resample(freq).apply(summarize_tides).unstack()


def summarize_tides(data: pd.Series) -> pd.Series:
    # used NOAA definitions - https://tidesandcurrents.noaa.gov/datum_options.html
    # also included spring and neap mean levels

    index = ["HAT", "MHHW", "MHW", "MTL", "MSL", "MLW", "MLLW", "LAT", "MN", "MSHW", "MSLW", "MNHW", "MNLW"]

    msl = data.mean()
    hat = data.max()
    lat = data.min()
    mhhw = data.resample("1D").max().mean()
    mllw = data.resample("1D").min().mean()

    highs, lows = find_pv(data=data, distance="8H")

    mhw = highs.mean()
    mlw = lows.mean()
    mn = mhw - mlw
    mtl = (mhw + mlw) / 2

    file = pkg_resources.open_text(__package__, resource="lunar_phases.csv")
    lunar_phases = (
        pd.read_csv(file, index_col="datetime", parse_dates=True).squeeze().loc[data.index[0] : data.index[-1]]
    )
    if data.index.tzinfo:
        tz = data.index.tzinfo

    springs = lunar_phases.loc[(lunar_phases == "New Moon") | (lunar_phases == "Full Moon")].tz_convert(tz)
    neaps = lunar_phases.loc[(lunar_phases == "First Quarter") | (lunar_phases == "Last Quarter")].tz_convert(tz)

    mshw = pd.concat([highs, pd.Series(data=np.nan, index=springs.index)]).sort_index()
    mshw = mshw.interpolate("time").loc[springs.index].mean()

    mslw = pd.concat([lows, pd.Series(data=np.nan, index=springs.index)]).sort_index()
    mslw = mslw.interpolate("time").loc[springs.index].mean()

    mnhw = pd.concat([highs, pd.Series(data=np.nan, index=neaps.index)]).sort_index()
    mnhw = mnhw.interpolate("time").loc[neaps.index].mean()

    mnlw = pd.concat([lows, pd.Series(data=np.nan, index=neaps.index)]).sort_index()
    mnlw = mnlw.interpolate("time").loc[neaps.index].mean()

    vals = [hat, mhhw, mhw, mtl, msl, mlw, mllw, lat, mn, mshw, mslw, mnhw, mnlw]

    return pd.Series(data=vals, index=index)
