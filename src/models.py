import numpy as np
import pandas as pd
import seaborn as sns
import time

from collections import namedtuple
from dataclasses import dataclass, field, InitVar
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from tqdm import tqdm

sns.set()


@dataclass
class TidalFlat:
    tide_ts: pd.Series
    land_elev_init: InitVar[float]
    conc_bound: float
    grain_diam: float
    grain_dens: float
    bulk_dens: float
    org_rate_yr: float = 0.0
    comp_rate_yr: float = 0.0
    sub_rate_yr: float = 0.0
    slr_yr: float = 0.0
    pos: int = 0
    year: int = 0
    aggr_total: float = 0.0
    degr_total: float = 0.0
    inundations: list = field(default_factory=list)
    results: list = field(default_factory=list)
    timestep: float = field(init=False)
    pbar: tqdm = field(init=False)
    runtime: float = None

    def __post_init__(self, land_elev_init):
        self.land_elev = land_elev_init
        self.timestep = self.tide_ts.index[1] - self.tide_ts.index[0]

    @staticmethod
    def stokes_settling(
        grain_diam, grain_dens, fluid_dens=1000.0, fluid_visc=0.001, g=9.8
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
        return (
            abs(self.org_rate_yr - self.comp_rate_yr - self.sub_rate_yr)
            / 365
            / 24
            / 60
            / 60
        )

    def make_subset(self, search_distance="day"):
        day = 60 * 60 * 24
        week = day * 7
        month = day * 30

        n = eval(search_distance)
        end = self.pos + n

        subset = self.tide_ts.loc[self.pos : end].to_frame(name="tide_elev")
        subset["land_elev"] = (
            self.land_elev - (subset.index - subset.index[0]) * self.linear_rate_sec
        )
        num_crossings = len(
            np.where(np.diff(np.signbit(subset.tide_elev - subset.land_elev)))[0]
        )

        count = 0
        while num_crossings < 2:
            if subset.index[-1] == self.tide_ts.index[-1] and num_crossings == 1:
                print("Warning: Subset finishes above platform.")
                return subset
            elif subset.index[-1] == self.tide_ts.index[-1] and num_crossings == 0:
                return subset
            else:
                end = end + n
                subset = self.tide_ts.loc[self.pos : end].to_frame(name="tide_elev")
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
                n = week
        return subset

    def find_inundation(self):
        subset = self.make_subset()
        if subset.index[-1] == self.tide_ts.index[-1]:
            return [subset, None]
        pos_start = (subset.tide_elev > subset.land_elev).idxmax()
        pos_end = (
            subset.loc[pos_start:].tide_elev > subset.loc[pos_start:].land_elev
        ).idxmin()
        assert pos_end > pos_start
        land_elev_init = subset.land_elev.loc[pos_start]
        inundation = Inundation(
            tide_ts=subset.tide_elev.loc[pos_start:pos_end],
            land_elev_init=land_elev_init,
            conc_bound=self.conc_bound,
            settle_rate=self.settle_rate,
            bulk_dens=self.bulk_dens,
            linear_rate_sec=self.linear_rate_sec,
            seed=self.pos,
        )
        return [subset.loc[self.pos : pos_start - self.timestep], inundation]

    def step(self):
        subset_before, inundation = self.find_inundation()
        if inundation:
            self.inundations.append(inundation)
            inundation.integrate()
        self.update(subset_before, inundation)

    def run(self, steps=np.inf):
        self._initialize(steps=steps)
        n = 0
        while self.pos < self.tide_ts.index[-1] and n < steps:
            self.step()
            n += 1
        self._unitialize()

    def _initialize(self, steps=None):
        self.runtime = time.perf_counter()
        if steps is not np.inf:
            pbar_total = steps
        else:
            pbar_total = int(len(self.tide_ts) / 60 / 60 / 24)
        self.pbar = tqdm(
            total=pbar_total, unit="day", position=0, leave=True, desc="Progress",
        )

    def _unitialize(self):
        self.runtime = time.perf_counter() - self.runtime
        self.pbar.close()
        self.print_results()

    def update(self, subset, inundation):
        # self.results = self.results.append([subset])
        self.results.append(subset)
        self.degr_total = self.degr_total + (
            subset.land_elev.values[0] - subset.land_elev.values[-1]
        )
        if inundation:
            self.degr_total = self.degr_total + inundation.degr_total
            self.aggr_total = self.aggr_total + inundation.aggr_total
            self.results.append(inundation.df[["tide_elev", "land_elev"]])
            self.land_elev = inundation.result.y[2][-1]
            self.pos = inundation.pos_end + self.timestep
            self.pbar.n = round(inundation.pos_end / 60 / 60 / 24)
            self.pbar.refresh()
        else:
            self.results = pd.concat(self.results)
            self.results = self.results.set_index(
                keys=pd.TimedeltaIndex(self.results.index, unit="s")
            )
            self.land_elev = subset.land_elev.values[-1]
            self.pos = subset.index[-1] + self.timestep
            self.pbar.n = round(subset.index[-1] / 60 / 60 / 24)
            self.pbar.refresh()
            del self.inundations

    def print_results(self):
        print("-" * 40)
        print(
            "{:<25} {:<10.3f} {:>2}".format(
                "Starting elevation: ", self.results.land_elev.iat[0], "m"
            )
        )
        print(
            "{:<25} {:<10.3f} {:>2}".format(
                "Final elevation: ", self.results.land_elev.iat[-1], "m"
            )
        )
        print(
            "{:<25} {:<10.3f} {:>2}".format(
                "Elevation change: ",
                (self.results.land_elev.iat[-1] - self.results.land_elev.iat[0]) * 100,
                "cm",
            )
        )
        print("-" * 40)
        print(
            "{:<25} {:<10.3f} {:>2}".format(
                "Aggradation: ", self.aggr_total * 100, "cm"
            )
        )
        print(
            "{:<25} {:<10.3f} {:>2}".format(
                "Degradation: ", self.degr_total * 100, "cm"
            )
        )
        print("-" * 40)
        print(
            "{:<25} {:>13}".format(
                "Runtime: ", time.strftime("%M min %S s", time.gmtime(self.runtime))
            )
        )

    def plot(self, frac=1.0):
        data = self.results.sample(frac=frac).sort_index()
        start = pd.to_datetime("01-01-2020") - self.results.index[0]
        end = pd.to_datetime("01-01-2020") - self.results.index[-1]
        data["time"] = pd.to_datetime("01-01-2020") - data.index
        plt.figure(figsize=(15, 5))
        ax = sns.lineplot(
            data=data,
            x="time",
            y="tide_elev",
            alpha=0.6,
            color="cornflowerblue",
            label="Tide Elevation",
        )
        sns.lineplot(
            ax=ax,
            data=data,
            x="time",
            y="land_elev",
            color="forestgreen",
            label="Land Elevation",
        )
        locator = mdates.MonthLocator(bymonth=(1, 7))
        formatter = mdates.DateFormatter("%b %Y")
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_xlim(start, end)


@dataclass
class Inundation:
    tide_ts: pd.Series
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
        self.pos_start = self.tide_ts.index[0]
        self.pos_end = self.tide_ts.index[-1]
        self.pos_slack = np.argmax(self.tide_ts.values) + self.pos_start
        self.tide_elev_slack = np.max(self.tide_ts.values)
        self.period = self.pos_end - self.pos_start
        tide_elev_func = InterpolatedUnivariateSpline(
            x=self.tide_ts.index.values, y=self.tide_ts.values, k=4,
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
        tide_elev = self.result.y[0]
        conc = self.result.y[1]
        land_elev = self.result.y[2]
        aggr = self.result.y[3]
        aggr_max = self.result.y[4]
        degr = self.result.y[5]
        df = pd.DataFrame(
            data={
                "tide_elev": tide_elev,
                "land_elev": land_elev,
                "conc": conc,
                "aggr": aggr,
                "aggr_max": aggr_max,
                "degr": degr,
            },
            index=time,
        )
        if self.result.status == 1:
            solver_end_time = self.result.t[-1]
            next_pos = int(np.ceil(solver_end_time))
            solver_end_diff = next_pos - solver_end_time
            small_degr = solver_end_diff * abs(self.params.linear_rate_sec)
            df2 = self.tide_ts.loc[next_pos:].to_frame(name="tide_elev")
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
        return y[1] - 1e-8

    zero_conc.terminal = True
    zero_conc.direction = -1

    zero_conc = staticmethod(zero_conc)

    def zero_depth(t, y, params):
        return y[0] - y[2] - 1e-8

    zero_depth.terminal = True
    zero_depth.direction = -1

    zero_depth = staticmethod(zero_depth)

    def integrate(self, method="DOP853", dense_output=True):

        self.result = solve_ivp(
            fun=self.solve_odes,
            t_span=[self.pos_start, self.pos_end],
            y0=[self.tide_ts.values[0], 0.0, self.land_elev_init, 0.0, 0.0, 0.0],
            method=method,
            events=(self.zero_conc, self.zero_depth),
            dense_output=dense_output,
            atol=(1e-6, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8),
            args=[self.params],
        )

        self.aggr_total = self.result.y[3][-1]
        self._set_df()
        self._validate_result()

    def _validate_result(self):
        assert self.result.success is True, "[t={}] Integration failed!".format(
            self.pos_start
        )
        assert (
            self.result.y[0] >= self.result.y[2]
        ).all(), "[t={}] Negative depths detected!\ndepths={}".format(
            self.pos_start, self.result.y[0] - self.result.y[2],
        )
        assert (
            self.result.y[1] >= 0.0
        ).all(), "[t={}] Negative concentrations detected!".format(self.pos_start,)
        if (self.result.y[4] >= self.result.y[3]).all() is False:
            where = np.where(self.result.y[4] <= self.result.y[3])
            assert np.allclose(
                a=self.result.y[4][where], b=self.result.y[3][where]
            ), "[t={}] Overextraction detected!".format(self.pos_start)

    def plot(self):

        fig, axs = plt.subplots(4, 1)
        fig.set_figheight(15)
        fig.set_figwidth(15)

        time = (self.df.index - self.df.index[0]) / 60
        mod_end = (self.result.t[-1] - self.df.index[0]) / 60
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
        axs[0].set(ylabel="Elevation (m)")
        axs[0].set(xticklabels=[])

        sns.lineplot(ax=axs[1], x=time, y=self.df.conc, color="saddlebrown")
        axs[1].set(ylabel="Concentration (g/L)",)
        axs[1].set(xticklabels=[])

        sns.lineplot(
            ax=axs[2], x=time, y=self.df.aggr_max, color="red", label="Max",
        )
        sns.scatterplot(
            ax=axs[2],
            x=np.append(time[time <= mod_end], time[time > mod_end][0::60]),
            y=np.append(
                self.df.aggr.values[time <= mod_end],
                self.df.aggr.values[time > mod_end][0::60],
            ),
            label="Modeled",
        )
        axs[2].set(ylabel="Aggradation (m)")
        axs[2].set(xticklabels=[])
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
                ((self.pos_slack - self.pos_start) / 60), color="black", linestyle="--",
            )
            ax.ticklabel_format(axis="y", useOffset=False)
