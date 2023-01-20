from __future__ import annotations

# from loguru import logger
import importlib.resources as pkg_resources
from copy import deepcopy
from dataclasses import InitVar, dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import detrend

from .constants import YEAR
from .helpers import find_pv
from .core.cycle import Cycle


@dataclass
class Tides:
    series: InitVar[pd.Series]
    data: pd.DataFrame = field(init=False)
    cycles: pd.DataFrame = field(init=False)
    datums: pd.DataFrame = field(init=False)

    @property
    def start(self) -> pd.Timestamp:
        return self.data.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.data.index[-1]

    @property
    def period(self) -> pd.Timedelta:
        return self.end - self.start

    @property
    def freq(self) -> pd.DateOffset:
        return self.data.index.freq

    @property
    def levels(self) -> pd.Series:
        return self.data.levels

    @property
    def lows(self) -> pd.Series:
        return self.levels[self.data.low]

    @property
    def highs(self) -> pd.Series:
        return self.levels[self.data.high]

    def __post_init__(self, series: pd.Series) -> None:
        highs, lows = find_pv(data=series, distance="8H")
        highs = highs.loc[lows.index[0] : lows.index[-1]]
        self.data = series.loc[lows.index[0] : lows.index[-1]].to_frame(name="levels").rename_axis(index="datetime")
        self.data["low"] = self.data.index.isin(lows.index)
        self.data["high"] = self.data.index.isin(highs.index)
        self.update()

    def update(self) -> None:
        self.cycles = self.describe_cycles()
        self.datums = self.summarize()

    def describe_cycles(self) -> pd.DataFrame:
        return pd.DataFrame(
            data={
                "start": self.lows.index[:-1],
                "slack": self.highs.index,
                "end": self.lows.index[1:],
                "period": self.lows.index[1:] - self.lows.index[:-1],
                "high": self.highs.values,
                "low1": self.lows.iloc[:-1].values,
                "low2": self.lows.iloc[1:].values,
                "range1": self.highs.values - self.lows.iloc[:-1].values,
                "range2": self.highs.values - self.lows.iloc[1:].values,
            },
        ).rename_axis("cycle")

    def raise_sea_level(self, slr: float) -> Tides:
        copy = deepcopy(self)
        copy.data.levels += slr * (self.data.index - self.start) / YEAR
        copy.update()
        return copy

    def amplify(self, factor: float) -> Tides:
        copy = deepcopy(self)
        detrended = detrend(copy.data.levels)
        trend = copy.data.levels - detrended
        copy.data.levels = detrended * factor + trend
        copy.update()
        return copy

    def subset(self, start: str | None = None, end: str | None = None) -> Tides:
        copy = deepcopy(self)
        i = copy.data.loc[start:end].low.idxmax()
        ii = copy.data.loc[start:end].low[::-1].idxmax()
        copy.data = copy.data.loc[i:ii]
        copy.update()
        return copy

    def make_cycle(self, n: int) -> Cycle:
        cycle = self.cycles.loc[n]
        return Cycle(cycle_n=n, water_levels=self.levels.loc[cycle.start : cycle.end])

    def summarize(self, freq: str | None = None, **kwargs) -> pd.Series:
        if freq:
            return self.levels.resample(freq, **kwargs).apply(summarize_tides).unstack()
        else:
            return summarize_tides(data=self.levels)


def summarize_tides(data: pd.Series) -> pd.Series:
    # used NOAA definitions - https://tidesandcurrents.noaa.gov/datum_options.html
    # also included spring and neap mean levels

    datums = ["HAT", "MHHW", "MHW", "MTL", "MSL", "MLW", "MLLW", "LAT", "MN", "MSHW", "MSLW", "MNHW", "MNLW"]

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

    file = pkg_resources.files(__package__).joinpath("data/lunar_phases.csv")
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

    return pd.Series(data=vals, index=datums).rename_axis("datums")
