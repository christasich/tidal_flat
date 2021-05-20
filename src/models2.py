from logging import setLogRecordFactory
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
            (self.org_rate_yr + self.comp_rate_yr + self.sub_rate_yr)
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
        self.timestep = np.int(pd.to_timedelta(tide_ts.index.freq).total_seconds())
        self.ts_len = len(tide_ts)
        self.tide_ts = pd.Series(
            data=tide_ts.values,
            index=pd.RangeIndex(start=0, stop=self.ts_len, step=self.timestep,),
        )
        self.land_elev = land_elev_init
        self.pos = self.tide_ts.index.values[0]
        self.year = 0
        self.remaining_inundations = len(self.zero_crossings()) / 2
        self.inundations = []
        self.results = pd.DataFrame(columns=["tide_elev", "land_elev"])

    def __repr__(self):
        return "{name} @{id:x} {attrs}".format(
            name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    def land_elev_degraded(self, pos_start=None, pos_end=None):
        if pos_start is None:
            pos_start = self.pos
        if pos_end is None:
            pos_end = self.tide_ts.index[-1]
        index = self.tide_ts.loc[pos_start:pos_end].index.values
        return self.land_elev + (index - index[0]) * self.linear_rate_sec

    @property
    def offset(self):
        return self.ts_len * self.year

    def zero_crossings(self):
        tide_elev = self.tide_ts[self.pos :].values
        land_elev = self.land_elev_degraded()
        depth = tide_elev - land_elev
        index = np.where(np.diff(np.signbit(depth)))[0]
        positions = np.where(depth[index] > 0, index, index + 1)
        crossings_pos = positions + self.pos
        assert (depth[positions] > 0).all()
        crossings_land_elev = land_elev[positions]
        crossings = np.array(
            list(zip(crossings_pos, crossings_land_elev)), dtype=object
        )
        return crossings

    def find_inundation(self):
        remaining_crossings = self.zero_crossings()
        self.remaining_inundations = len(remaining_crossings) / 2
        while True:
            if self.remaining_inundations == 0:
                return None

            pos_start = remaining_crossings[0][0]
            land_elev_init = remaining_crossings[0][1]
            if self.remaining_inundations == 0.5:
                print("Warning: Partial inundation cycle!")
                pos_end = self.tide_ts.index[-1]
            elif self.remaining_inundations >= 1:
                pos_end = remaining_crossings[1][0]

            if pos_end - pos_start < 4:
                remaining_crossings = remaining_crossings[2:]
                print("Small inundation at t={}->{}. Skipping.")
                continue
            inundation = Inundation(
                tide_ts=self.tide_ts[pos_start:pos_end],
                land_elev_init=land_elev_init,
                conc_bound=self.conc_bound,
                settle_rate=self.settle_rate,
                bulk_dens=self.bulk_dens,
                linear_rate_sec=self.linear_rate_sec,
            )

            return inundation

    def inundate(self, inundation):
        inundation.integrate()
        self.inundations.append(inundation)
        self.land_elev = inundation.df.land_elev.values[-1]

    def step(self):
        inundation = self.find_inundation()
        if inundation:
            self.inundate(inundation)
        self.update(inundation)

    def run(self, steps=np.inf):
        n = 0
        pbar = tqdm(
            leave=True,
            position=0,
            unit="inundation",
            desc="Progress",
            total=self.remaining_inundations,
        )
        while self.pos < self.ts_len:
            if n == steps:
                return
            self.step()
            pbar.total = pbar.n + self.remaining_inundations - 1
            pbar.update()
            n += 1
        self.year += 1
        # self.update()

    def update(self, inundation):
        if inundation:
            df1 = self.tide_ts.loc[self.pos : inundation.pos_start].to_frame(
                name="tide_elev"
            )
            df1["land_elev"] = self.land_elev_degraded(
                pos_start=self.pos, pos_end=inundation.pos_start
            )
            df2 = inundation.df[["tide_elev", "land_elev"]]
            self.results = pd.concat([self.results, df1, df2])
            self.pos = inundation.pos_end + self.timestep * 2
        else:
            df = self.tide_ts.loc[self.pos :].to_frame(name="tide_elev")
            df["land_elev"] = self.land_elev_degraded(pos_start=self.pos)
            self.results = pd.concat([self.results, df])
            self.pos = self.tide_ts.index[-1] + self.timestep

    # def update(self):
    #     self.tide_ts = self.tide_ts + self.slr_yr
    #     self.tide_ts.index = self.tide_ts.index + self.offset


class Inundation:
    def __init__(
        self,
        tide_ts,
        land_elev_init,
        conc_bound,
        settle_rate,
        bulk_dens,
        linear_rate_sec,
    ):
        self.tide_ts = tide_ts
        self.land_elev_init = land_elev_init
        self.pos_start = tide_ts.index[0]
        self.pos_end = tide_ts.index[-1]
        self.pos_slack = np.argmax(self.tide_ts.values) + self.pos_start
        self.tide_elev_slack = np.max(self.tide_ts.values)
        self.period = self.pos_end - self.pos_start
        tide_elev_func = InterpolatedUnivariateSpline(
            x=self.tide_ts.index.values, y=self.tide_ts.values, k=4, ext=2,
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
        self.flood = None
        self.ebb = None
        self.result = None
        self.df = None
        self._res_tup = namedtuple("result", ["t", "y"])

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
            solver_end_pos = int(np.ceil(solver_end_time))
            solver_end_diff = solver_end_pos - solver_end_time
            df2 = self.tide_ts.loc[solver_end_pos:].to_frame(name="tide_elev")
            remaining_time = df2.index.values - df2.index[0]
            df2["conc"] = 0.0
            df2["aggr"] = df.aggr.values[-1]
            df2["aggr_max"] = df.aggr_max.values[-1]
            degr = (
                abs(self.params.linear_rate_sec) * solver_end_diff
                + abs(self.params.linear_rate_sec) * remaining_time
            )
            df2["land_elev"] = df.land_elev.values[-1] - degr
            df2["degr"] = df.degr.values[-1] + degr
            df = pd.concat([df, df2])

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
        if d1dt_depth > 0.0:
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
            d1dt_degr,  # 5
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
            fun=self.solve_odes,
            t_span=[self.pos_start, self.pos_slack],
            y0=[self.tide_ts.values[0], 0.0, self.land_elev_init, 0.0, 0.0, 0.0],
            method=method,
            dense_output=dense_output,
            args=[self.params],
        )

        self.ebb = solve_ivp(
            fun=self.solve_odes,
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
            dense_output=dense_output,
            events=(self.zero_depth, self.zero_conc),
            args=[self.params],
        )
        self.result = self._res_tup(
            t=np.append(self.flood.t, self.ebb.t[1:]),
            y=np.append(self.flood.y, self.ebb.y[:, 1:], axis=1),
        )

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
        assert np.isclose(
            self.result.y[5][-1],
            (self.result.t[-1] - self.result.t[0]) * abs(self.params.linear_rate_sec),
        ), "{} - {}".format(
            self.result.y[5][-1],
            (self.result.t[-1] - self.result.t[0]) * abs(self.params.linear_rate_sec),
        )

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
