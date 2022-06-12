from dataclasses import InitVar, dataclass, field

import numpy as np
import pandas as pd
import seaborn as sns
import utide
from loguru import logger
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from utide.utilities import Bunch

from . import constants
from .utils import find_pv, lowess_ts, make_trend, regress_ts, exponential, quadratic


def load_tides(path: str) -> pd.Series:
    """
    Function that loads the tidal curve constructed by make_tides.R
    and sets the index to the Datetime column and infers frequency.
    """
    data = pd.read_feather(path)
    vals = data.elevation.values
    index = pd.DatetimeIndex(data.datetime, freq="infer")
    tides = pd.Series(data=vals, index=index, name="elevation")
    return tides


def model_tides(
    data: pd.Series,
    lat: float,
    start: str,
    end: str,
    freq: str,
    conf_int: str = "MC",
    method: str = "ols",
    trend: bool = False,
    nodal: bool = False,
    constit: str = "auto",
    verbose: bool = False,
) -> Bunch:

    u = data.values
    t = mdates.date2num(data.index.to_pydatetime())
    coef = utide.solve(
        t=t, u=u, lat=lat, conf_int=conf_int, method=method, trend=trend, nodal=nodal, constit=constit, verbose=verbose
    )

    index = pd.date_range(start=start, end=end, freq=freq)
    t_predict = mdates.date2num(index.to_pydatetime())
    predicted = utide.reconstruct(t=t_predict, coef=coef, verbose=verbose)

    return pd.Series(data=predicted["h"], index=index)


def flag_extrema(data: pd.Series, include_roll: bool = False) -> pd.DataFrame:

    data = data.to_frame(name="elevation")

    # Find highs and lows
    highs, lows = find_pv(data=data.elevation, window="8H")
    data["high"] = data.index.isin(highs.index)
    data["low"] = data.index.isin(lows.index)

    # Find amplitude from rolling highs and lows
    high_roll = data.elevation.loc[data.high].rolling(window=2, center=True).mean()
    low_roll = data.elevation.loc[data.low].rolling(window=2, center=True).mean()
    roll = pd.concat([high_roll, low_roll], keys=["high_roll", "low_roll"], axis=1)
    roll = roll.resample("10T").mean().interpolate(limit_area="inside")
    roll["tidal_range"] = roll.high_roll - roll.low_roll

    # Find springs and neaps
    springs, neaps = find_pv(data=roll.tidal_range, window="11D")
    data["spring"] = data.index.isin(springs.index)
    data["neap"] = data.index.isin(neaps.index)

    if include_roll:
        return pd.concat([data, roll], axis=1)
    else:
        return data


def summarize_tides(data: pd.Series) -> pd.Series:
    # used NOAA definitions - https://tidesandcurrents.noaa.gov/datum_options.html
    # also included spring and neap mean levels

    data = flag_extrema(data=data, include_roll=True)

    index = ["HAT", "MHHW", "MHW", "MTL", "MSL", "MLW", "MLLW", "LAT", "MN", "MSHW", "MSLW", "MNHW", "MNLW"]

    msl = data.elevation.mean()
    hat = data.elevation.max()
    lat = data.elevation.min()
    mhhw = data.elevation.resample("1D").max().mean()
    mllw = data.elevation.resample("1D").min().mean()
    mhw = data.loc[data.high].elevation.mean()
    mlw = data.loc[data.low].elevation.mean()
    mn = mhw - mlw
    mtl = (mhw + mlw) / 2
    mshw = data.loc[data.spring].high_roll.mean()
    mslw = data.loc[data.spring].low_roll.mean()
    mnhw = data.loc[data.neap].high_roll.mean()
    mnlw = data.loc[data.neap].low_roll.mean()

    vals = [hat, mhhw, mhw, mtl, msl, mlw, mllw, lat, mn, mshw, mslw, mnhw, mnlw]

    return pd.Series(data=vals, index=index)


@dataclass
class Tides:
    water_levels: InitVar[pd.Series]
    data: pd.DataFrame = field(init=False)

    def __post_init__(self, water_levels: pd.Series) -> None:

        self.data = flag_extrema(data=water_levels, include_roll=True)
        self.summary = summarize_tides(data=water_levels)

    @property
    def highs(self) -> pd.Series:
        return self.data.loc[self.data.high].elevation

    @property
    def lows(self) -> pd.Series:
        return self.data.loc[self.data.low].elevation

    @property
    def springs(self) -> pd.DataFrame:
        springs = self.data.loc[self.data.spring][["high_roll", "low_roll"]]
        springs.columns = ["highs", "lows"]
        return springs

    @property
    def neaps(self) -> pd.DataFrame:
        neaps = self.data.loc[self.data.neap][["high_roll", "low_roll"]]
        neaps.columns = ["highs", "lows"]
        return neaps

    def calc_amplification(
        self,
        beta: float | tuple[float, float] | pd.Series | tuple[pd.Series, pd.Series],
        k: float | int,
        benchmarks: tuple[str, str] = ["MHW", "MLW"],
    ) -> pd.Series:

        t = (self.data.index - self.data.index[0]).values / pd.Timedelta("365.25D")
        if isinstance(beta, int | float):
            beta_high = k * (exponential(t=t, a=beta, k=k) - beta)
            beta_low = -beta_high
        elif isinstance(beta, tuple | list):
            beta_high = k * (exponential(t=t, a=beta, k=k) - beta)
            beta_low = k * (exponential(t=t, a=beta, k=k) - beta)

        bench_high = self.summary[benchmarks[0]]
        bench_low = self.summary[benchmarks[1]]

        cond = self.data.elevation > self.summary.MSL

        above = (self.data.elevation - self.summary.MSL) / (bench_high - self.summary.MSL) * beta_high
        below = (self.data.elevation - self.summary.MSL) / (bench_low - self.summary.MSL) * beta_low

        return below.mask(cond=cond, other=above)

    def calc_slr(self, z2100=None, a=None, b=None):
        if z2100:
            index = pd.date_range(start="2000", end=self.data.index[-1], freq=self.data.index.freq)
            t = (index - index[0]).values / pd.Timedelta("365.25D")
            a = (z2100 - b * 100) / 100 ** 2
            slr = pd.Series(index=index, data=quadratic(t=t, a=a, b=b)).loc[self.data.index[0]:]
            slr = slr - slr.iat[0]
        else:
            t = (self.data.index - self.data.index[0]).values / pd.Timedelta("365.25D")
            slr = pd.Series(index=self.data.index, data=quadratic(t=t, a=a, b=b))
        return slr

    def plot(self, start: str = None, end: str = None, freq: str = None) -> None:
        if freq is None:
            freq = self.data.index.freq

        subset = self.data.loc[start:end]

        _, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)

        sns.lineplot(data=subset.elevation.resample(rule=freq).mean(), ax=ax, color="cornflowerblue", alpha=0.5)
        sns.scatterplot(
            data=subset.loc[subset.high].reset_index(),
            x="datetime",
            y="elevation",
            ax=ax,
            color="green",
            s=15,
            zorder=15,
        )
        sns.scatterplot(
            data=subset.loc[subset.low].reset_index(),
            x="datetime",
            y="elevation",
            ax=ax,
            color="red",
            s=15,
            zorder=15,
        )
        sns.lineplot(data=subset.mw.resample(rule=freq).mean(), ax=ax, color="black", alpha=0.5)
        sns.lineplot(data=subset.mhw.resample(rule=freq).mean(), ax=ax, color="green", alpha=0.5)
        sns.lineplot(data=subset.mlw.resample(rule=freq).mean(), ax=ax, color="red", alpha=0.5)

        for loc in self.springs.loc[start:end].index:
            ax.axvline(x=loc, color="black", linestyle="dotted", alpha=0.5)
            ax.text(x=loc, y=ax.get_ylim()[1], s="S", rotation=0, ha="center", va="bottom")

        for loc in self.neaps.loc[start:end].index:
            ax.axvline(x=loc, color="black", linestyle="dotted", alpha=0.5)
            ax.text(x=loc, y=ax.get_ylim()[1], s="N", rotation=0, ha="center", va="bottom")

        ax.set_xlim(subset.index[0], subset.index[-1])

    def lm_plot(self, vars: list = ["SH", "H", "NH", "NL", "L", "SL"], s: int = 10) -> None:

        names = ["SH", "H", "NH", "NL", "L", "SL"]
        series = [self.springs.highs, self.highs, self.neaps.highs, self.neaps.lows, self.lows, self.springs.lows]
        colors = ["black", "green", "black", "black", "red", "black"]
        markers = ["^", ".", "X", "X", ".", "^"]
        rowColors = ["lightgreen", "lightgreen", "lightgreen", "pink", "pink", "pink"]
        s = np.array([3, 1, 3, 3, 1, 3]) * s
        zorder = [20, 10, 20, 20, 10, 20]
        alpha = [0.6, 1, 0.6, 0.6, 1, 0.6]

        vals = [
            {
                "ts": ts,
                "color": color,
                "marker": marker,
                "rowColor": rowColor,
                "s": s,
                "zorder": zorder,
                "alpha": alpha,
            }
            for ts, color, marker, rowColor, s, zorder, alpha in zip(
                series, colors, markers, rowColors, s, zorder, alpha
            )
        ]
        df = pd.DataFrame(dict(zip(names, vals)))[vars]

        ref_date = self.data.index[0]
        freq = pd.Timedelta("365.25D")

        for var in vars:
            df.loc["lm", var], df.loc["coef", var], df.loc["intercept", var] = regress_ts(
                ts=df[var].ts, ref_date=ref_date, freq=freq
            )

        fig, (ax, tabax) = plt.subplots(
            figsize=(13, 5), ncols=2, gridspec_kw={"width_ratios": [6, 1]}, constrained_layout=True
        )

        sns.lineplot(
            data=self.data.resample("1H").mean().reset_index(),
            x="datetime",
            y="elevation",
            ax=ax,
            color="cornflowerblue",
            alpha=0.6,
        )

        x = self.data.index[[0, -1]]
        X = ((x - ref_date) / freq).values.reshape(-1, 1)
        for name, data in df.iteritems():
            y = data.lm.predict(X).flatten()
            sns.scatterplot(
                data=data.ts,
                ax=ax,
                marker=data.marker,
                color=data.color,
                s=data.s,
                linewidth=0,
                alpha=data.alpha,
                zorder=data.zorder,
            )
            sns.lineplot(x=x, y=y, ls="--", color="black", ax=ax, zorder=30)
            ax.text(
                x=self.data.index[-1],
                y=y[-1],
                ha="left",
                va="center",
                zorder=30,
                s=name,
                fontsize="medium",
                fontweight="bold",
            )

        colLabels = [r"$\Delta\zeta\ mm\ yr^{-1}$"]
        colColors = ["lightgray"]
        cellText = (df.loc["coef"] * 1000).astype(float).round(decimals=1).values.reshape(-1, 1)
        rowLabels = df.columns
        rowColors = df.loc["rowColor"]

        tabax.axis("off")
        tabax.table(
            cellText=cellText,
            colLabels=colLabels,
            rowLabels=rowLabels,
            rowLoc="center",
            cellLoc="center",
            loc="center",
            colColours=colColors,
            rowColours=rowColors,
        )
        ax.xaxis.label.set_visible(False)
        ax.set_xlim(self.data.index[0], self.data.index[-1])
