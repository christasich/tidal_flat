from typing import NamedTuple
import numpy as np
import sys
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import time

from collections import namedtuple
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass, field, InitVar
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from scipy.signal import argrelextrema, find_peaks
from tqdm import tqdm
import utide

from .utils import *

def model_tides(tides: pd.Series, lat: float, start: str, end: str, freq:str, conf_int:str="MC", method:str="ols", trend: bool=False, nodal: bool=False, constit: str="auto", verbose: bool=False):

    u = tides.values
    t = mdates.date2num(tides.index.to_pydatetime())
    coef = utide.solve(t=t, u=u, lat=lat, conf_int=conf_int, method=method, trend=trend, nodal=nodal, constit=constit, verbose=verbose)

    index = pd.date_range(start=start, end=end, freq=freq)
    t_predict = mdates.date2num(index.to_pydatetime())
    model = utide.reconstruct(t=t_predict, coef=coef, verbose=verbose)

    return model

def describe_tides(data: pd.Series):

    df = data.to_frame(name="elevation")
    df["base"] = df.elevation
    df["years"] = ((df.index - df.index[0]) / pd.Timedelta("365.25D")).values

    df = df[["years", "elevation", "base"]]

    df[["high", "low", "spring", "neap"]] = False

    df["mw"] = lowess_ts(data=df.base.groupby(by=pd.Grouper(freq=pd.Timedelta("12H25T"))).mean(), window=pd.Timedelta("12H25T")*6).reindex_like(df).interpolate(limit_direction="both")

    # Find highs and lows
    HL_dist = pd.Timedelta("1H") / df.index.freq * 8
    highs, lows = find_pv(x=df.base, distance=HL_dist)
    df.loc[df.iloc[highs].index, "high"] = True
    df.loc[df.iloc[lows].index, "low"] = True

    # Find amplitude from smoothed highs and lows
    df["mhw"] = lowess_ts(data=df.query("high==True").base, window=pd.Timedelta("12H25T")*6).reindex_like(df).interpolate(limit_direction="both")
    df["mlw"] = lowess_ts(data=df.query("low==True").base, window=pd.Timedelta("12H25T")*6).reindex_like(df).interpolate(limit_direction="both")
    df["mtr"] = df.mhw - df.mlw

    # Find springs and neaps
    SN_dist = pd.Timedelta("1H") / df.index.freq * 24 * 11
    springs, neaps = find_pv(x=df.mtr, distance=SN_dist)
    df.loc[df.iloc[springs].index, "spring"] = True
    df.loc[df.iloc[neaps].index, "neap"] = True

    df["mshw"] = lowess_ts(data=df.query("spring==True").mhw, window=pd.Timedelta("365.25D")).reindex_like(df).interpolate(limit_direction="both")
    df["mslw"] = lowess_ts(data=df.query("spring==True").mlw, window=pd.Timedelta("365.25D")).reindex_like(df).interpolate(limit_direction="both")

    df["mnhw"] = lowess_ts(data=df.query("neap==True").mhw, window=pd.Timedelta("365.25D")).reindex_like(df).interpolate(limit_direction="both")
    df["mnlw"] = lowess_ts(data=df.query("neap==True").mlw, window=pd.Timedelta("365.25D")).reindex_like(df).interpolate(limit_direction="both")

    return df

@dataclass
class Tides:
    water_levels: InitVar[pd.Series]
    data: pd.DataFrame = field(init=False)

    def __post_init__(self, water_levels):

        self.data = describe_tides(data=water_levels)
        self.set_amplifier()

    @property
    def highs(self):
        return self.data.query("high==True").elevation

    @property
    def lows(self):
        return self.data.query("low==True").elevation

    @property
    def springs(self):
        springs = self.data.query("spring==True")[["mhw", "mlw"]]
        springs.columns = ["highs", "lows"]
        return springs

    @property
    def neaps(self):
        neaps = self.data.query("neap==True")[["mhw", "mlw"]]
        neaps.columns = ["highs", "lows"]
        return neaps

    def set_amplifier(self, method="spring"):
        if method=="spring":
            above = ((self.data.base - self.data.mw) / (self.data.mshw - self.data.mw) * self.data.years).loc[self.data.base > self.data.mw]
            below = ((self.data.base - self.data.mw) / (self.data.mslw - self.data.mw) *  self.data.years).loc[self.data.base < self.data.mw]
            self.data["A"] = pd.concat([above, below]).sort_index()

    def calc_trend(self, trend):
        return(trend * self.data.years)

    def calc_beta(self, beta):
        if isinstance(beta, float):
            beta_low = -beta
            beta_high = beta
        elif isinstance(beta, tuple):
            beta_low = beta[0]
            beta_high = beta[1]
        above = self.data.A.loc[self.data.base > self.data.mw] * beta_high
        below = self.data.A.loc[self.data.base < self.data.mw] * beta_low
        return(pd.concat([above, below]).sort_index())

    def calc_elevation(self, beta=0.0, trend=0.0):
        if beta != 0.0:
            beta = self.calc_beta(beta)
        if trend != 0.0:
            trend = self.calc_trend(trend)
        return(self.data.elevation + beta + trend)

    def plot(self, start=None, end=None, freq=None, show_change=False):
        if freq is None:
            freq = self.data.index.freq

        subset = self.data.loc[start:end]

        fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)

        if show_change is False:
            sns.lineplot(data=subset.elevation.resample(rule=freq).mean(), ax=ax, color="cornflowerblue", alpha=0.5)
            sns.scatterplot(data=subset.query("high==True").reset_index(), x="datetime", y="elevation", ax=ax, color="green", s=15, zorder=15)
            sns.scatterplot(data=subset.query("low==True").reset_index(), x="datetime", y="elevation", ax=ax, color="red", s=15, zorder=15)
        else:
            sns.lineplot(data=subset.elevation.resample(rule=freq).mean(), ax=ax, color="red", ls="--", alpha=0.5)
            sns.lineplot(data=subset.base.resample(rule=freq).mean(), ax=ax, color="cornflowerblue", alpha=0.5)
            err = subset.elevation - subset.base
            ax.errorbar(x=subset.query("high==True").index, y=subset.query("high==True").base.values, yerr=err.loc[subset.high==True].values, capsize=2, lolims=True, ls="none", color="green", zorder=1)
            ax.errorbar(x=subset.query("low==True").index, y=subset.query("low==True").base.values, yerr=err.loc[subset.low==True].values * -1, capsize=2,uplims=True, ls="none", color="red", zorder=1)
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

        return(fig, ax)
    
    def lm_plot(self, vars=["SH", "H", "NH", "NL", "L", "SL"], s=10):

        names = ["SH", "H", "NH", "NL", "L", "SL"]
        series =  [self.springs.highs, self.highs, self.neaps.highs, self.neaps.lows, self.lows, self.springs.lows]
        colors = ["black", "green", "black", "black", "red", "black"]
        markers = ["^", ".", "X", "X", ".", "^"]
        rowColors = ["lightgreen", "lightgreen", "lightgreen", "pink", "pink", "pink"]
        s = np.array([3, 1, 3, 3, 1, 3]) * s
        zorder = [20, 10, 20, 20, 10, 20]
        alpha = [0.6, 1, 0.6, 0.6, 1, 0.6]

        vals = [{'ts': ts, 'color': color, "marker": marker, "rowColor": rowColor, "s": s, "zorder": zorder, "alpha": alpha} for ts, color, marker, rowColor, s, zorder, alpha in zip(series, colors, markers, rowColors, s, zorder, alpha)]
        df = pd.DataFrame(dict(zip(names, vals)))[vars]

        ref_date = self.data.index[0]
        freq = pd.Timedelta("365.25D")

        for var in vars:
            df.loc["lm", var], df.loc["coef", var], df.loc["intercept", var] = regress_ts(ts=df[var].ts, ref_date=ref_date, freq=freq)

        fig, (ax, tabax) = plt.subplots(figsize=(13, 5), ncols=2, gridspec_kw={'width_ratios': [6, 1]}, constrained_layout=True)

        sns.lineplot(data=self.data.resample("1H").mean().reset_index(), x="datetime", y="elevation", ax=ax, color="cornflowerblue", alpha=0.6)

        x = self.data.index[[0, -1]]
        X = ((x - ref_date) / freq).values.reshape(-1, 1)
        for name, data in df.iteritems():
            y = data.lm.predict(X).flatten()
            sns.scatterplot(data=data.ts, ax=ax, marker=data.marker, color=data.color, s=data.s, linewidth=0, alpha=data.alpha, zorder=data.zorder)
            sns.lineplot(x=x, y=y, ls="--", color="black", ax=ax, zorder=30)
            ax.text(x=self.data.index[-1], y=y[-1], ha="left", va="center", zorder=30, s=name, fontsize="medium", fontweight="bold");

        colLabels = ["$\Delta\zeta\ mm\ yr^{-1}$"]
        colColors=["lightgray"]
        cellText = (df.loc["coef"] * 1000).astype(float).round(decimals=1).values.reshape(-1, 1)
        rowLabels = df.columns
        rowColors = df.loc["rowColor"]

        tabax.axis("off")
        tabax.table(cellText=cellText, colLabels=colLabels, rowLabels=rowLabels, rowLoc="center", cellLoc="center", loc='center', colColours=colColors, rowColours=rowColors)
        ax.xaxis.label.set_visible(False);
        ax.set_xlim(self.data.index[0], self.data.index[-1]);