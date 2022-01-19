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

from src.definitions import *

def find_pv(x: np.ndarray | pd.Series, distance: int):

    peaks = find_peaks(x=x, distance=distance)[0]
    valleys = find_peaks(x=x*-1, distance=distance)[0]

    return(peaks, valleys)

def ts_lm(ts: pd.Series, freq: str | pd.Timedelta, ref_date: pd.Timestamp):

    if isinstance(freq, str):
        freq = pd.Timedelta(freq)

    x = ((ts.index - ref_date) / freq).values.reshape(-1, 1)
    y = ts.values.reshape(-1, 1)
    lm = LinearRegression().fit(x, y)

    return(lm, lm.coef_[0,0], lm.intercept_[0])

def smooth_series(data: pd.Series, window: pd.Timedelta=None):
    endog = data.values
    exog = (data.index - data.index[0]).total_seconds().astype(int).values
    n = data.groupby(by=pd.Grouper(freq=window)).count().mean().round()
    frac = n / len(data)
    y = sm.nonparametric.lowess(endog=endog, exog=exog, frac=frac, is_sorted=True)[:,1]
    return(pd.Series(data=y, index=data.index))

def describe_tides(data: pd.Series):

    df = data.to_frame(name="elevation")
    df["base"] = df.elevation
    df["years"] = ((df.index - df.index[0]) / pd.Timedelta("365.25D")).values

    df = df[["years", "elevation", "base"]]

    df[["high", "low", "spring", "neap"]] = False

    df["mw"] = smooth_series(data=df.base.groupby(by=pd.Grouper(freq=pd.Timedelta("12H25T"))).mean(), window=pd.Timedelta("12H25T")*6).reindex_like(df).interpolate(limit_direction="both")

    # Find highs and lows
    HL_dist = pd.Timedelta("1H") / df.index.freq * 8
    highs, lows = find_pv(x=df.base, distance=HL_dist)
    df.loc[df.iloc[highs].index, "high"] = True
    df.loc[df.iloc[lows].index, "low"] = True

    # Find amplitude from smoothed highs and lows
    df["mhw"] = smooth_series(data=df.query("high==True").base, window=pd.Timedelta("12H25T")*6).reindex_like(df).interpolate(limit_direction="both")
    df["mlw"] = smooth_series(data=df.query("low==True").base, window=pd.Timedelta("12H25T")*6).reindex_like(df).interpolate(limit_direction="both")
    df["mtr"] = df.mhw - df.mlw

    # Find springs and neaps
    SN_dist = pd.Timedelta("1H") / df.index.freq * 24 * 11
    springs, neaps = find_pv(x=df.mtr, distance=SN_dist)
    df.loc[df.iloc[springs].index, "spring"] = True
    df.loc[df.iloc[neaps].index, "neap"] = True

    df["mshw"] = smooth_series(data=df.query("spring==True").mhw, window=pd.Timedelta("365.25D")).reindex_like(df).interpolate(limit_direction="both")
    df["mslw"] = smooth_series(data=df.query("spring==True").mlw, window=pd.Timedelta("365.25D")).reindex_like(df).interpolate(limit_direction="both")

    df["mnhw"] = smooth_series(data=df.query("neap==True").mhw, window=pd.Timedelta("365.25D")).reindex_like(df).interpolate(limit_direction="both")
    df["mnlw"] = smooth_series(data=df.query("neap==True").mlw, window=pd.Timedelta("365.25D")).reindex_like(df).interpolate(limit_direction="both")

    return df

@dataclass
class Tides:
    data: pd.Series
    trend: InitVar[float] = 0.0
    beta: InitVar[float | tuple[float,float]] = 0.0

    def __post_init__(self, trend, beta):

        self.data = describe_tides(data=self.data)
        self.set_A()

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

    def set_A(self, method="spring"):
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
            df.loc["lm", var], df.loc["coef", var], df.loc["intercept", var] = ts_lm(ts=df[var].ts, ref_date=ref_date, freq=freq)

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

@dataclass
class TidalFlat:
    tides: InitVar[pd.Series]
    land_elev_init: InitVar[float]
    conc_bound: float
    grain_diam: float
    grain_dens: float
    bulk_dens: float
    org_rate_yr: float = 0.0
    comp_rate_yr: float = 0.0
    sub_rate_yr: float = 0.0
    pos: int = 0
    aggr_total: float = 0.0
    degr_total: float = 0.0
    inundations: list = field(default_factory=list)
    results: list = field(default_factory=list)
    sim_length: pd.Timedelta = field(init=False)
    timestep: float = field(init=False)
    pbar: tqdm = field(init=False)
    pbar_pos: int = 0
    verbose: bool = False
    runtime: float = None

    def __post_init__(self, tides, land_elev_init):
        index = pd.date_range(
            start=tides.index[0]-pd.DateOffset(months=1),
            end=tides.index[-1]+pd.DateOffset(months=1),
            freq="MS"
            ) + pd.DateOffset(days=14)
        if not isinstance(self.conc_bound, float):
            assert len(self.conc_bound) == 12, "Concentration must be float or array of length 12."
            self.conc_bound = pd.Series(
                data=self.conc_bound[index.month-1],
                index=index
                ).asfreq("1D").interpolate().loc[tides.index[0]:tides.index[-1]]
        else:
            self.conc_bound = pd.Series(
                data=self.conc_bound[index.month-1],
                index=index
                ).asfreq("1D").interpolate().loc[tides.index[0]:tides.index[-1]]
        
        
        self.timestep = tides.index.freq.delta.total_seconds()
        self.sim_length = tides.index[-1] - tides.index[0]
        self.land_elev = land_elev_init
        index = pd.RangeIndex(
            start=0,
            stop=len(tides) * self.timestep,
            step=self.timestep,
            name="elapsed_sec",
        )
        self.tides = pd.DataFrame(
            data={"datetime": tides.index, "tide_elev": tides.values}, index=index
        )

    @staticmethod
    def stokes_settling(
        grain_diam,
        grain_dens,
        fluid_dens=WATER_DENSITY,
        fluid_visc=WATER_VISCOSITY,
        g=GRAVITY,
    ):
        settle_rate = (
            (2 / 9 * (grain_dens - fluid_dens) / fluid_visc) * g * (grain_diam / 2) ** 2
        )
        return settle_rate

    @property
    def settle_rate(self):
        return self.stokes_settling(
            grain_diam=self.grain_diam, grain_dens=self.grain_dens,
        )

    @property
    def linear_rate_sec(self):
        return abs(self.org_rate_yr - self.comp_rate_yr - self.sub_rate_yr) / YEAR

    def make_subset(self):            

        n = DAY
        end = self.pos + n

        subset = self.tides.loc[self.pos:end].copy()
        subset["land_elev"] = (
            self.land_elev - (subset.index - subset.index[0]) * self.linear_rate_sec
        )        
        num_crossings = len(
            np.where(np.diff(np.signbit(subset.tide_elev - subset.land_elev)))[0]
        )

        count = 0
        while num_crossings < 2:
            if subset.index[-1] == self.tides.index[-1] and num_crossings == 1:
                # print("Warning: Subset finishes above platform.")
                return subset
            elif subset.index[-1] == self.tides.index[-1] and num_crossings == 0:
                return subset
            else:
                end = end + n
                subset = self.tides.loc[self.pos:end].copy()
                subset["land_elev"] = (
                    self.land_elev
                    - (subset.index - subset.index[0]) * self.linear_rate_sec
                )
                num_crossings = len(
                    np.where(np.diff(np.signbit(subset.tide_elev - subset.land_elev)))[
                        0
                    ]
                )
            count += 1
            if count > 7:
                n = WEEK
            elif count > 7 + 2:
                n = MONTH
        return subset

    def find_inundation(self):
        # status: 
        #  -2 = initial tide above platform
        #  -1 = skipping inundation since len < 3
        #   0 = full inundation cycle
        #   1 = end of tidal data
        subset = self.make_subset()
        if subset.index[-1] == self.tides.index[-1]:
            return [subset, None, 1]

        if self.pos == 0 and subset.tide_elev.loc[self.pos] > self.land_elev:
            pos_start = 0
        else:
            pos_start = (subset.tide_elev > subset.land_elev).idxmax()

        pos_end = (
            subset.loc[pos_start:].tide_elev > subset.loc[pos_start:].land_elev
        ).idxmin()
        assert pos_end > pos_start

        if (pos_end - pos_start) / self.timestep < 3:
            return [subset.loc[self.pos:pos_end], None, -1]
        land_elev_init = subset.land_elev.loc[pos_start]
        inundation = Inundation(
            tides=subset.loc[pos_start:pos_end],
            land_elev_init=land_elev_init,
            conc_bound=self.conc_bound[self.tides.loc[pos_start].datetime.round("1D")],
            settle_rate=self.settle_rate,
            bulk_dens=self.bulk_dens,
            linear_rate_sec=self.linear_rate_sec,
            seed=self.pos,
        )
        if self.pos == 0 and subset.tide_elev.loc[self.pos] > self.land_elev:
            return [None, inundation, -2]
        else:
            return [subset.loc[self.pos:pos_start - self.timestep], inundation, 0]

    def step(self):
        subset_before, inundation, status = self.find_inundation()
        if status == 0 or status == -2:
            # self.inundations.append(inundation)
            inundation.integrate()
        self.update(subset_before, inundation, status)

    def run(self):
        self._initialize()
        while self.pos < self.tides.index[-1]:
            self.step()
            self.pbar.n = round((self.pos-self.timestep) / DAY)
            self.pbar.set_postfix({"Date": self.tides.loc[self.pos-self.timestep].datetime.strftime("%Y-%m-%d")})
        self._unitialize()

    def _initialize(self):
        
        self.runtime = time.perf_counter()
        postfix = {"Date": self.tides.datetime.iat[0].strftime("%Y-%m-%d")}
        self.pbar = tqdm(
            total=self.sim_length.ceil("D").days, unit="Day", position=self.pbar_pos, leave=True, postfix=postfix
        )

    def _unitialize(self):
        self.runtime = time.perf_counter() - self.runtime
        self.pbar.close()
        if self.verbose is True:
            self.print_results()

    def update(self, subset, inundation, status):
        if status == 0:
            self.results.append(subset)
            self.degr_total = self.degr_total + (
            subset.land_elev.values[0] - subset.land_elev.values[-1]
        )
            self.degr_total = self.degr_total + inundation.degr_total
            self.aggr_total = self.aggr_total + inundation.aggr_total
            self.results.append(inundation.df[["datetime", "tide_elev", "land_elev"]])
            self.land_elev = inundation.result.y[2][-1]
            self.pos = inundation.pos_end + self.timestep
        elif status == -1:
            self.results.append(subset)
            self.degr_total = self.degr_total + (
            subset.land_elev.values[0] - subset.land_elev.values[-1]
        )
            self.land_elev = subset.land_elev.values[-1]
            self.pos = subset.index[-1] + self.timestep
        elif status == 1:
            self.results.append(subset)
            self.degr_total = self.degr_total + (
            subset.land_elev.values[0] - subset.land_elev.values[-1]
        )
            self.results = pd.concat(self.results)
            self.land_elev = subset.land_elev.values[-1]
            self.pos = subset.index[-1] + self.timestep
        elif status == -2:
            self.degr_total = self.degr_total + inundation.degr_total
            self.aggr_total = self.aggr_total + inundation.aggr_total
            self.results.append(inundation.df[["datetime", "tide_elev", "land_elev"]])
            self.land_elev = inundation.result.y[2][-1]
            self.pos = inundation.pos_end + self.timestep

    def print_results(self):
        years = self.sim_length / pd.Timedelta("365.25D")
        print("{:<25} {:>10} {:>10} {:>5}".format("", "Mean Yearly", "Total", "Unit"))
        print("-" * 55)
        print("{:<25} {:>10} {:>10.3f} {:>5}".format("Starting elevation: ", "", self.results.land_elev.iat[0], "m"))
        print("{:<25} {:>10} {:>10.3f} {:>5}".format("Final elevation: ", "", self.results.land_elev.iat[-1], "m"))
        print("{:<25} {:>10.3f} {:>10.3f} {:>5}".format("Elevation change: ", (self.results.land_elev.iat[-1] - self.results.land_elev.iat[0]) * 100 / years, (self.results.land_elev.iat[-1] - self.results.land_elev.iat[0]) * 100,"cm",))
        print("-" * 55)
        print("{:<25} {:>10.3f} {:>10.3f} {:>5}".format("Aggradation: ", self.aggr_total * 100 / years, self.aggr_total * 100, "cm"))
        print("{:<25} {:>10.3f} {:>10.3f} {:>5}".format("Degradation: ", self.degr_total * 100 / years, self.degr_total * 100, "cm"))
        print("-" * 55)
        print("{:<25} {:>25}".format("Runtime: ", time.strftime("%M min %S s", time.gmtime(self.runtime))))

    def plot(self, frac=1.0):
        data = self.results.sample(frac=frac)

        fig, ax1 = plt.subplots(figsize=(15, 5), constrained_layout=True)
        ax2 = ax1.twinx()
        sns.lineplot(
            ax=ax1,
            x=data.datetime,
            y=data.tide_elev,
            alpha=0.6,
            color="cornflowerblue",
            label="Tide Elevation",
            legend=False,
        )
        sns.lineplot(
            ax=ax2,
            x=data.datetime,
            y=data.land_elev - self.results.land_elev.iat[0],
            color="forestgreen",
            label="Land Elevation",
            legend=False,
        )
        ax1.set(
            xlim=(self.results.datetime.iat[0], self.results.datetime.iat[-1]),
            ylim=(self.results.land_elev.min(), self.results.tide_elev.max()),
            xlabel="Year",
            ylabel="Elevation (m)",
        )
        ax2.set(
            xlim=(self.results.datetime.iat[0], self.results.datetime.iat[-1]),
            ylim=(
                self.results.land_elev.min() - self.results.land_elev.iat[0],
                self.results.tide_elev.max() - self.results.land_elev.iat[0],
            ),
            ylabel="$\Delta$ Elevation (m)",
        )
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2)
        return(fig, [ax1, ax2])


@dataclass
class Inundation:
    tides: pd.DataFrame
    land_elev_init: float
    conc_bound: InitVar[float]
    settle_rate: InitVar[float]
    bulk_dens: InitVar[float]
    linear_rate_sec: InitVar[float]
    seed: int
    result: OdeResult = None
    df: pd.DataFrame = None
    aggr_total: float = None
    degr_total: float = None

    def __post_init__(self, conc_bound, settle_rate, bulk_dens, linear_rate_sec):
        self.timestep = self.tides.index[1] - self.tides.index[0]
        self.pos_start = self.tides.index[0]
        self.pos_end = self.tides.index[-1]
        self.pos_slack = np.argmax(self.tides.tide_elev.values) + self.pos_start
        self.tide_elev_slack = np.max(self.tides.tide_elev.values)
        self.period = self.pos_end - self.pos_start
        tide_elev_func = InterpolatedUnivariateSpline(
            x=self.tides.index.values, y=self.tides.tide_elev.values, k=3,
        )
        params = namedtuple(
            "params",
            [
                "tide_elev_func",
                "conc_bound",
                "settle_rate",
                "bulk_dens",
                "linear_rate_sec",
            ],
        )
        self.params = params(
            tide_elev_func=tide_elev_func,
            conc_bound=conc_bound,
            settle_rate=settle_rate,
            bulk_dens=bulk_dens,
            linear_rate_sec=linear_rate_sec,
        )

    def _set_df(self):
        time = self.result.t
        time_diff = time - time[0]
        datetime = np.array(
            [self.tides.datetime.iat[0] + pd.Timedelta(i, unit="s") for i in time_diff]
        )
        tide_elev = self.result.y[0]
        conc = self.result.y[1]
        land_elev = self.result.y[2]
        aggr = self.result.y[3]
        aggr_max = self.result.y[4]
        degr = self.result.y[5]
        df = pd.DataFrame(
            data={
                "datetime": datetime,
                "tide_elev": tide_elev,
                "land_elev": land_elev,
                "conc": conc,
                "aggr": aggr,
                "aggr_max": aggr_max,
                "degr": degr,
            },
            index=time,
        )
        df.index.rename(name="elapsed_sec", inplace=True)
        if self.result.status == 1:
            solver_end_time = self.result.t[-1]
            next_pos = int(np.ceil(solver_end_time))
            solver_end_diff = next_pos - solver_end_time
            small_degr = solver_end_diff * abs(self.params.linear_rate_sec)
            df2 = self.tides.loc[next_pos:].copy()
            df2["land_elev"] = df.land_elev.values[-1]
            df2["conc"] = 0.0
            df2["aggr"] = df.aggr.values[-1]
            df2["aggr_max"] = df.aggr_max.values[-1]
            df2["degr"] = (
                df.degr.values[-1]
                + small_degr
                + abs(self.params.linear_rate_sec) * (df2.index - df2.index[0])
            )
            df = pd.concat([df, df2])

        self.degr_total = df.degr.values[-1]
        self.df = df

    @staticmethod
    def solve_odes(
        t, y, params,
    ):
        tide_elev = y[0]
        conc = y[1]
        land_elev = y[2]
        depth = tide_elev - land_elev

        d1dt_tide_elev = params.tide_elev_func.derivative()(t)
        d1dt_aggr = params.settle_rate * conc / params.bulk_dens
        d1dt_degr = abs(params.linear_rate_sec)
        d1dt_land_elev = d1dt_aggr - d1dt_degr
        d1dt_depth = d1dt_tide_elev - d1dt_land_elev
        if d1dt_depth > 0:
            d1dt_conc = (
                -(params.settle_rate * conc) / depth
                - 1 / depth * (conc - params.conc_bound) * d1dt_depth
            )
            d1dt_aggr_max = params.conc_bound * d1dt_depth / params.bulk_dens
        else:
            d1dt_conc = -(params.settle_rate * conc) / depth
            d1dt_aggr_max = 0.0

        return [
            d1dt_tide_elev,  # 0
            d1dt_conc,  # 1
            d1dt_land_elev,  # 2
            d1dt_aggr,  # 3
            d1dt_aggr_max,  # 4
            d1dt_degr,
        ]

    def zero_conc(t, y, params):
        return y[1] - 1e-5

    zero_conc.terminal = True
    zero_conc.direction = -1

    zero_conc = staticmethod(zero_conc)

    def zero_depth(t, y, params):
        return y[0] - y[2] - 1e-5

    zero_depth.terminal = True
    zero_depth.direction = -1

    zero_depth = staticmethod(zero_depth)

    def integrate(self, method="DOP853", dense_output=True):

        self.result = solve_ivp(
            fun=self.solve_odes,
            t_span=[self.pos_start, self.pos_end],
            y0=[
                self.tides.tide_elev.values[0],
                0.0,
                self.land_elev_init,
                0.0,
                0.0,
                0.0,
            ],
            method=method,
            events=(self.zero_conc, self.zero_depth),
            dense_output=dense_output,
            atol=(1e-6, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8),
            args=[self.params],
        )

        self.aggr_total = self.result.y[3][-1]
        self._set_df()
        # self._validate_result()

    def _validate_result(self):
        assert self.result.success is True, "[t={}] Integration failed!\nparams={}".format(self.pos_start, self.params)
        assert (self.result.y[3] >= 0.0).all(), "[t={}] Negative aggradation detected!\naggr={}".format(self.pos_start, self.results.y[3])
        # if ~((self.result.y[4] - self.result.y[3]) >= 0).all():
        #     print("[t={}] Overextraction detected!\nDifference={}".format(self.pos_start, (self.result.y[4] - self.result.y[3])))
        # assert ((self.result.y[4] - self.result.y[3]) >= -1e-6).all(), "[t={}] Overextraction detected!".format(self.pos_start)

        # if (self.result.y[4] >= self.result.y[3]).all() is False:
        #     where = np.where(self.result.y[4] <= self.result.y[3])
        #     assert np.allclose(
        #         a=self.result.y[4][where], b=self.result.y[3][where]
        #     ), "[t={}] Overextraction detected!".format(self.pos_start)

    def plot(self):

        fig, axs = plt.subplots(nrows=4, ncols=1, tight_layout=True)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        fig.suptitle("Inundation at {}".format(self.tides.datetime.iat[0]), fontsize=16)

        time = (self.df.index - self.df.index[0]) / MINUTE
        mod_end = (self.result.t[-1] - self.df.index[0]) / MINUTE
        aggr_max_mod_diff = self.df.aggr_max - self.df.aggr

        sns.lineplot(
            ax=axs[0],
            x=time,
            y=self.df.tide_elev,
            color="cornflowerblue",
            label="Tide",
        )
        sns.lineplot(
            ax=axs[0], x=time, y=self.df.land_elev, color="forestgreen", label="Land"
        )
        axs[0].set(xlabel="", ylabel="Elevation (m)", xticklabels=[])

        sns.lineplot(ax=axs[1], x=time, y=self.df.conc, color="saddlebrown")
        axs[1].set(xlabel="", ylabel="Concentration (g/L)", xticklabels=[])

        sns.lineplot(
            ax=axs[2], x=time, y=self.df.aggr_max, color="red", label="Max",
        )
        sns.scatterplot(
            ax=axs[2],
            x=np.append(time[time <= mod_end], time[time > mod_end][0::MINUTE]),
            y=np.append(
                self.df.aggr.values[time <= mod_end],
                self.df.aggr.values[time > mod_end][0::MINUTE],
            ),
            label="Modeled",
        )
        axs[2].set(xlabel="", ylabel="Aggradation (m)", xticklabels=[])
        axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        sns.lineplot(
            ax=axs[3], x=time, y=aggr_max_mod_diff, color="black", linestyle=":"
        )
        axs[3].set(ylabel="Difference (m)\nMax - Modeled", xlabel="Time (min)")
        axs[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axs[3].fill_between(
            x=time,
            y1=aggr_max_mod_diff,
            where=aggr_max_mod_diff >= 0.0,
            color="forestgreen",
            alpha=0.3,
        )
        axs[3].fill_between(
            x=time,
            y1=aggr_max_mod_diff,
            where=aggr_max_mod_diff < 0,
            color="red",
            alpha=0.3,
        )
        for ax in axs:
            ax.axvline(
                ((self.pos_slack - self.pos_start) * self.timestep / MINUTE),
                color="black",
                linestyle="--",
            )
            ax.ticklabel_format(axis="y", useOffset=False)

def simulate(tides: Tides, params: NamedTuple):
    
    tf = TidalFlat(
        tides=tides.calc_elevation(beta=params.beta, trend=params.slr),
        land_elev_init=params.land_elev_init,
        conc_bound=params.conc_bound.values,
        grain_diam=params.grain_diam,
        grain_dens=2.65e3,
        bulk_dens=params.bulk_dens,
        org_rate_yr=2e-4,
        comp_rate_yr=4e-3,
        sub_rate_yr=3e-3,
        pbar_pos=params.n,
    )
    tf.run()

    name = "result_z-{:.2f}_conc-{:.2f}_gs-{:.1e}_bd-{}.feather".format(params.land_elev_init, params.conc_bound.max(), params.grain_diam, params.bulk_dens)
    path = r"/home/chris/projects/tidal_flat_0d/data/results" + name
    tf.results.reset_index().to_feather(path)