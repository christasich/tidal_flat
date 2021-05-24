from logging import setLogRecordFactory
from termios import N_MOUSE
import numpy as np
import pandas as pd
import seaborn as sns
import time

from collections import namedtuple
from matplotlib import pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from tqdm import tqdm

sns.set()


class TidalFlat:
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

    def __init__(
        self,
        tide_ts,
        land_elev_init,
        conc_bound,
        grain_diam,
        grain_dens,
        bulk_dens,
        org_rate_yr=0.0,
        comp_rate_yr=0.0,
        sub_rate_yr=0.0,
        slr_yr=0.0,
    ):
        self.conc_bound = conc_bound
        self.grain_diam = grain_diam
        self.grain_dens = grain_dens
        self.bulk_dens = bulk_dens
        self.org_rate_yr = org_rate_yr
        self.comp_rate_yr = comp_rate_yr
        self.sub_rate_yr = sub_rate_yr
        self.slr_yr = slr_yr
        self.tide_ts = tide_ts
        self.land_elev = land_elev_init
        self.pos = 0
        self.year = 0
        self.inundations = []
        self.pbar = tqdm(
            total=int(len(self.tide_ts) / 60 / 60 / 24),
            unit="day",
            position=0,
            leave=True,
            desc="Progress",
        )

    def __repr__(self):
        return "{name} @{id:x} {attrs}".format(
            name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    def make_subset(self):
        day = 60 * 60 * 24
        week = day * 7
        month = day * 30
        search_distance = day
        end = self.pos + search_distance
        subset = self.tide_ts.loc[self.pos : end].to_frame(name="tide_elev")
        subset["land_elev"] = (
            self.land_elev - (subset.index - subset.index[0]) * self.linear_rate_sec
        )
        num_crossings = len(
            np.where(np.diff(np.signbit(subset.tide_elev - subset.land_elev)))[0]
        )
        while num_crossings < 2:
            if subset.index[-1] == self.tide_ts.index[-1] and num_crossings == 1:
                print("Warning: Subset finishes above platform.")
                return subset
            elif subset.index[-1] == self.tide_ts.index[-1] and num_crossings == 0:
                return None
            else:
                end = end + search_distance
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
        return subset

    def find_inundation(self):
        subset = self.make_subset()
        if subset is None:
            return None
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
        return inundation

    def step(self):
        inundation = self.find_inundation()
        if inundation:
            self.inundations.append(inundation)
            inundation.integrate()
            self.update(inundation)
        else:
            self.land_elev = (
                self.land_elev
                - (self.tide_ts.index[-1] - self.pos) * self.linear_rate_sec
            )
            self.pos = self.tide_ts.index[-1]
            self.pbar.n = int(self.pos / 60 / 60 / 24 + 1)
            self.pbar.refresh()
            self.pbar.close()

    def run(self, steps=np.inf):
        n = 0
        while self.pos < self.tide_ts.index[-1]:
            if n == steps:
                return
            self.step()
            n += 1

    def update(self, inundation):
        self.land_elev = inundation.result.y[2][-1]
        self.pos = inundation.pos_end + 1
        self.pbar.n = int(self.pos / 60 / 60 / 24 + 1)
        self.pbar.refresh()


class OdeResults:
    def __init__(
        self, t, y,
    ):
        self.t = t
        self.y = y


class Inundation:
    def __init__(
        self,
        tide_ts,
        land_elev_init,
        conc_bound,
        settle_rate,
        bulk_dens,
        linear_rate_sec,
        seed,
    ):
        self.tide_ts = tide_ts
        self.land_elev_init = land_elev_init
        self.pos_start = tide_ts.index[0]
        self.pos_end = tide_ts.index[-1]
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
        self.result = None
        self.df = None
        self.seed = seed
        self.aggr_total = None
        self.degr_total = None

    def __repr__(self):
        return "{name} @{id:x} {attrs}".format(
            name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
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
        if self.ebb.status == 1:
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
    def solve_flood(
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
        d1dt_conc = (
            -(params.settle_rate * conc) / depth
            - 1 / depth * (conc - params.conc_bound) * d1dt_depth
        )
        d1dt_aggr_max = params.conc_bound * d1dt_depth / params.bulk_dens

        return [
            d1dt_tide_elev,  # 0
            d1dt_conc,  # 1
            d1dt_land_elev,  # 2
            d1dt_aggr,  # 3
            d1dt_aggr_max,  # 4
            d1dt_degr,
        ]

    @staticmethod
    def solve_ebb(
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
        return y[1]

    zero_conc.terminal = True
    zero_conc.direction = -1

    zero_conc = staticmethod(zero_conc)

    def zero_depth(t, y, params):
        return y[0] - y[2]

    zero_depth.terminal = True
    zero_depth.direction = -1

    zero_depth = staticmethod(zero_depth)

    def integrate(self, method="DOP853", dense_output=True):

        self.flood = solve_ivp(
            fun=self.solve_flood,
            t_span=[self.pos_start, self.pos_slack],
            y0=[self.tide_ts.values[0], 0.0, self.land_elev_init, 0.0, 0.0, 0.0],
            method=method,
            events=(self.zero_conc, self.zero_depth),
            dense_output=dense_output,
            atol=(1e-6, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8),
            args=[self.params],
        )

        self.ebb = solve_ivp(
            fun=self.solve_ebb,
            t_span=[self.pos_slack, self.pos_end],
            y0=[
                self.tide_elev_slack,
                self.flood.y[1][-1],
                self.flood.y[2][-1],
                self.flood.y[3][-1],
                self.flood.y[4][-1],
                self.flood.y[5][-1],
            ],
            method=method,
            events=(self.zero_conc, self.zero_depth),
            dense_output=dense_output,
            atol=(1e-6, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8),
            args=[self.params],
        )

        self.result = OdeResults(
            t=np.append(self.flood.t, self.ebb.t[1:]),
            y=np.append(self.flood.y, self.ebb.y[:, 1:], axis=1),
        )

        self.aggr_total = self.result.y[3][-1]
        self._set_df()
        self._validate_result()

    def _validate_result(self):
        assert self.flood.success is True, "[t={}] Flood integration failed!".format(
            self.pos_start
        )
        assert self.ebb.success is True, "[t={}] Ebb integration failed!".format(
            self.pos_slack
        )
        assert (
            self.result.y[0] >= self.result.y[2]
        ).all(), "[t={}] Negative depths detected!\ndepths={}".format(
            self.pos_start, self.result.y[0] - self.result.y[2],
        )
        assert (
            self.result.y[4] >= self.result.y[3]
        ).all(), "[t={}] Overextraction detected!".format(self.pos_start)

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
