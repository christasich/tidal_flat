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


# def apply_rate(init_val, index_per_sec, timestep_sec, rate_per_sec):
#     weights = np.arange(0, len(index_per_sec), 1) * timestep_sec
#     vals = init_val + (weights * rate_per_sec)
#     return vals


def apply_rate(init_val, length, rate_per_step):
    weights = np.arange(0, length) * rate_per_step
    vals = init_val + weights
    return vals


def stokes_settling(grain_diam, grain_dens, fluid_dens=1000.0, fluid_visc=0.001, g=9.8):
    """
    Function that uses Stokes' Law to calculate the settling velocity of a spherical particle
    falling through a fluid column.
    """
    settle_rate = (
        (2 / 9 * (grain_dens - fluid_dens) / fluid_visc) * g * (grain_diam / 2) ** 2
    )
    return settle_rate


def aggrade_flood(
    t, y, args,
):
    tide_elev = y[0]
    conc = y[1]
    land_elev = y[2]
    depth = tide_elev - land_elev

    d1dt_tide_elev = args.tide_func.derivative()(t)
    d1dt_land_elev = args.settle_rate * conc / args.bulk_dens
    d1dt_depth = d1dt_tide_elev - d1dt_land_elev
    d1dt_conc = (
        -(args.settle_rate * conc) / depth
        - 1 / depth * (conc - args.conc_bound) * d1dt_depth
    )
    d1dt_agg_max = args.conc_bound * d1dt_depth / args.bulk_dens

    return [
        d1dt_tide_elev,
        d1dt_conc,
        d1dt_land_elev,
        d1dt_agg_max,
    ]


def aggrade_ebb(
    t, y, args,
):

    tide_elev = y[0]
    conc = y[1]
    land_elev = y[2]
    depth = tide_elev - land_elev

    d1dt_tide_elev = args.tide_func.derivative()(t)
    d1dt_land_elev = args.settle_rate * conc / args.bulk_dens
    d1dt_conc = -(args.settle_rate * conc) / depth
    d1dt_agg_max = 0.0

    return [
        d1dt_tide_elev,
        d1dt_conc,
        d1dt_land_elev,
        d1dt_agg_max,
    ]


def zero_conc(t, y, args):
    return y[1]


zero_conc.terminal = True
zero_conc.direction = -1


def zero_depth(t, y, args):
    return y[0] + args.min_depth


zero_depth.terminal = True
zero_depth.direction = -1


def tide_df(index=None, tide_elev=None, land_elev=None, conc=None):
    if all(arg is None for arg in [index, tide_elev, land_elev, conc]):
        df = pd.DataFrame(columns=["tide_elev", "conc", "land_elev"])
    elif conc is None:
        df = pd.DataFrame(
            data={"tide_elev": tide_elev, "land_elev": land_elev}, index=index
        )
    else:
        df = pd.DataFrame(
            data={"tide_elev": tide_elev, "conc": conc, "land_elev": land_elev},
            index=index,
        )
    return df


class Params:
    def __init__(
        self,
        timestep=None,
        conc_init=None,
        conc_bound=None,
        grain_diam=None,
        grain_dens=None,
        bulk_dens=None,
        org_rate=None,
        comp_rate=None,
        sub_rate=None,
        slr=0.0,
        method="DOP853",
        min_depth=0.0,
        dense_output=False,
    ):
        self.timestep = timestep
        self.conc_init = conc_init
        self.conc_bound = conc_bound
        self.grain_diam = grain_diam
        self.grain_dens = grain_dens
        self.bulk_dens = bulk_dens
        self.org_rate = org_rate
        self.comp_rate = comp_rate
        self.sub_rate = sub_rate
        self.slr = slr
        self.method = method
        self.min_depth = min_depth
        self.dense_output = dense_output

    def __repr__(self):
        return "{name} @{id:x} {attrs}".format(
            name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    @property
    def settle_rate(self):
        return stokes_settling(
            grain_diam=self.grain_diam,
            grain_dens=self.grain_dens,
            fluid_dens=1000.0,
            fluid_visc=0.001,
            g=9.8,
        )

    @property
    def linear_rate_sec(self):
        return (self.org_rate + self.comp_rate + self.sub_rate) / 365 / 24 / 60 / 60


class LimbResult(OdeResult):
    def __init__(self, result, limb, params, pos_start, pos_end):
        super().__init__(result)
        self.limb = limb
        self.params = params
        self.pos_start = pos_start
        self.pos_end = pos_end
        self._validate

    def __repr__(self):
        return "{name} @{id:x} {attrs}".format(
            name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    def _validate(self):
        assert self.success is True, "[t={}] {} integration failed!".format(
            self.t[0], self.limb
        )
        assert (
            self.tide_elev >= self.land_elev
        ).all(), "[t={}] Negative depths detected for {} limb.\ndepths={}".format(
            self.t[0], self.limb, self.tide_elev - self.land_elev,
        )
        assert (self.agg_max >= self.agg).all(), "Overextraction on {}".format(
            self.limb
        )

    @property
    def time(self):
        return self.t

    @property
    def tide_elev(self):
        return self.y[0]

    @property
    def conc(self):
        return self.y[1]

    @property
    def land_elev(self):
        return self.y[2]

    @property
    def agg(self):
        return self.y[2] - self.y[2][0]

    @property
    def agg_max(self):
        return self.y[3]

    @property
    def agg_total(self):
        return self.agg[-1]

    @property
    def deg_total(self):
        return (self.pos_end - self.pos_start) * self.params.linear_rate_sec

    @property
    def df(self):
        return tide_df(
            index=self.time,
            tide_elev=self.tide_elev,
            land_elev=self.land_elev,
            conc=self.conc,
        )


class InundationResult:
    def __init__(self, flood=None, ebb=None):
        self.flood = flood
        self.ebb = ebb
        self.df = None

    def __repr__(self):
        return "{name} @{id:x} {attrs}".format(
            name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    @property
    def time(self):
        return np.append(self.flood.time, self.ebb.time[1:])

    @property
    def tide_elev(self):
        return np.append(self.flood.tide_elev, self.ebb.tide_elev[1:])

    @property
    def conc(self):
        return np.append(self.flood.conc, self.ebb.conc[1:])

    @property
    def land_elev(self):
        return np.append(self.flood.land_elev, self.ebb.land_elev[1:])

    @property
    def agg(self):
        return np.append(self.flood.agg, (self.ebb.agg + self.flood.agg[-1])[1:])

    @property
    def agg_max(self):
        return np.append(
            self.flood.agg_max, (self.ebb.agg_max + self.flood.agg[-1])[1:]
        )

    def set_df(self):
        self.df = tide_df(
            index=self.time,
            tide_elev=self.tide_elev,
            land_elev=self.land_elev,
            conc=self.conc,
        )

    @property
    def slack_time(self):
        return self.ebb.t[0]

    @property
    def agg_total(self):
        return self.flood.agg_total + self.ebb.agg_total

    @property
    def deg_total(self):
        return self.flood.deg_total + self.ebb.deg_total

    def plot(self, limb="both"):

        fig, axs = plt.subplots(4, 1)
        fig.set_figheight(15)
        fig.set_figwidth(15)

        if limb == "both":
            time = (self.time - self.time[0]) / 60
            tide_elev = self.tide_elev
            land_elev = self.land_elev
            agg = self.agg
            agg_max = self.agg_max
            conc = self.conc
            for ax in axs:
                ax.axvline(
                    ((self.slack_time - self.time[0]) / 60),
                    color="black",
                    linestyle="--",
                )
        else:
            time = eval("self.{}.time".format(limb))
            time = (time - time[0]) / 60
            tide_elev = eval("self.{}.tide_elev".format(limb))
            land_elev = eval("self.{}.land_elev".format(limb))
            agg = eval("self.{}.agg".format(limb))
            agg_max = eval("self.{}.agg_max".format(limb))
            conc = eval("self.{}.conc".format(limb))

        agg_max_mod_diff = agg_max - agg

        sns.lineplot(ax=axs[0], x=time, y=tide_elev, color="blue", label="Tide")
        sns.lineplot(ax=axs[0], x=time, y=land_elev, color="green", label="Land")
        axs[0].set(ylabel="Elevation (m)")
        axs[0].set(xticklabels=[])

        sns.lineplot(ax=axs[1], x=time, y=conc, color="red")
        axs[1].set(ylabel="Concentration (g/L)",)
        axs[1].set(xticklabels=[])

        sns.lineplot(
            ax=axs[2], x=time, y=agg_max, color="red", label="Max",
        )
        sns.lineplot(
            ax=axs[2],
            x=time,
            y=agg,
            label="Modeled",
            marker="o",
            linestyle=(0, (1, 10)),
        )
        axs[2].set(ylabel="Aggradation (m)")
        axs[2].set(xticklabels=[])
        axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        sns.lineplot(ax=axs[3], x=time, y=agg_max_mod_diff, color="black")
        axs[3].set(ylabel="Difference (m)\nMax - Modeled", xlabel="Time (min)")
        axs[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axs[3].fill_between(
            x=time,
            y1=agg_max_mod_diff,
            where=agg_max_mod_diff >= 0.0,
            color="green",
            alpha=0.3,
        )
        axs[3].fill_between(
            x=time,
            y1=agg_max_mod_diff,
            where=agg_max_mod_diff < 0,
            color="red",
            alpha=0.3,
        )

        for ax in axs:
            ax.ticklabel_format(axis="y", useOffset=False)


class Inundation:
    def __init__(self, index, tide_elev, land_elev_init, params, seed=None):
        self.index = index
        self.tide_elev = tide_elev
        self.land_elev_init = land_elev_init
        self.params = params
        self.seed = seed
        self.result = None

    def __repr__(self):
        return "{name} @{id:x} {attrs}".format(
            name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    @property
    def tide_elev_func(self):
        return InterpolatedUnivariateSpline(x=self.index, y=self.tide_elev, k=4, ext=2)

    @property
    def pos_start(self):
        return self.index[0]

    @property
    def pos_slack(self):
        return np.argmax(self.tide_elev) + self.index[0]

    @property
    def pos_end(self):
        return self.index[-1]

    @property
    def period(self):
        return self.index[-1]

    @property
    def tide_elev_slack(self):
        return np.max(self.tide_elev)

    @property
    def solver_args(self):
        args = namedtuple(
            "aggrade_args",
            ["tide_func", "conc_bound", "settle_rate", "bulk_dens", "min_depth"],
        )
        return args(
            tide_func=self.tide_elev_func,
            conc_bound=self.params.conc_bound,
            settle_rate=self.params.settle_rate,
            bulk_dens=self.params.bulk_dens,
            min_depth=self.params.min_depth,
        )

    def aggrade(self):
        self.result = InundationResult()

        # Integrate flood limb
        flood = solve_ivp(
            fun=aggrade_flood,
            t_span=[self.pos_start, self.pos_slack],
            y0=[self.tide_elev[0], self.params.conc_init, self.land_elev_init, 0.0,],
            method=self.params.method,
            dense_output=self.params.dense_output,
            args=[self.solver_args],
        )
        flood = LimbResult(
            result=flood,
            limb="flood",
            params=self.params,
            pos_start=self.pos_start,
            pos_end=self.pos_slack,
        )

        self.result.flood = flood

        # Integrate ebb limb
        ebb = solve_ivp(
            fun=aggrade_ebb,
            t_span=[self.pos_slack, self.pos_end],
            y0=[
                self.tide_elev_slack,
                self.result.flood.conc[-1],
                self.result.flood.land_elev[-1],
                self.result.flood.agg_max[-1] - self.result.flood.agg[-1],
            ],
            events=(zero_conc, zero_depth),
            method=self.params.method,
            dense_output=self.params.dense_output,
            args=[self.solver_args],
        )
        ebb = LimbResult(
            result=ebb,
            limb="ebb",
            params=self.params,
            pos_start=self.pos_slack,
            pos_end=self.pos_end,
        )

        self.result.ebb = ebb
        self.result.set_df()


class Tides:
    def __init__(self, tide_ts=None, land_elev_init=None, params=None):
        self.index_offset = tide_ts.index.values[0]
        self.index = tide_ts.index.values - self.index_offset
        self.tide_elev = tide_ts.values
        self.land_elev = land_elev_init + (self.index) * params.linear_rate_sec
        self.params = params
        self.pos = self.index[0]
        self.remaining_inundations = len(self.zero_crossings()) / 2

    def __repr__(self):
        return "{name} @{id:x} {attrs}".format(
            name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    def zero_crossings(self, pos=None):
        if pos is None:
            pos = self.pos
        crossings = (
            np.where(
                np.diff(
                    np.signbit(
                        self.tide_elev[pos:]
                        - (self.land_elev[pos:] + self.params.min_depth)
                    )
                )
            )[0]
            + 1
        )
        return crossings + pos

    def find_inundation(self):
        remaining_crossings = self.zero_crossings()
        self.remaining_inundations = len(remaining_crossings) / 2
        while True:
            if self.remaining_inundations == 0:
                return None
            elif self.remaining_inundations == 0.5:
                print("Warning: Partial inundation cycle!")
                start = remaining_crossings[0]
                end = self.index[-1]
            elif self.remaining_inundations >= 1:
                start, end = remaining_crossings[0:2]
                end = end

            if end - start < 4:
                remaining_crossings = remaining_crossings[2:]
                print("Small inundation at t={}->{}. Skipping.")
                continue

            assert (
                self.tide_elev[start:end]
                > self.land_elev[start:end] + self.params.min_depth
            ).all(), "start={}, end={}, seed={}".format(start, end, self.pos)

            inundation = Inundation(
                index=self.index[start:end] + self.index_offset,
                tide_elev=self.tide_elev[start:end],
                land_elev_init=self.land_elev[start],
                params=self.params,
                seed=self.pos,
            )

            return inundation

    def update(self, inundation: Inundation):
        if inundation is None:
            self.pos = self.index[-1]
            return
        before = self.land_elev[: inundation.pos_end - self.index_offset]
        after = (
            inundation.result.land_elev[-1]
            + (
                self.index[inundation.pos_end - self.index_offset :]
                - self.index[inundation.pos_end - self.index_offset]
            )
            * self.params.linear_rate_sec
        )
        new = np.append(before, after)

        assert len(new) == len(self.land_elev)

        self.land_elev = new
        self.pos = inundation.pos_end - self.index_offset + self.params.timestep


class SimulationResults:
    def __init__(self):
        self.inundations = []
        self.agg_total = 0.0
        self.deg_total = 0.0
        self.df = tide_df()

    def __repr__(self):
        return "{name} @{id:x} {attrs}".format(
            name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    @property
    def land_elev_change(self):
        return self.df.land_elev.iat[-1] - self.df.land_elev.iat[0]

    def update(self, tide: Tides, inundation: Inundation):
        if inundation is None:
            tide_end = tide.index[-1] + tide.index_offset
            inundation_result = None
        else:
            self.inundations.append(inundation)
            inundation_result = inundation.result.df
            tide_end = inundation.pos_start
            self.agg_total = self.agg_total + inundation.result.agg_total
            self.deg_total = self.deg_total + inundation.result.deg_total

        tide_result = tide_df(
            index=tide.index[
                tide.pos - tide.index_offset : tide_end - tide.index_offset
            ],
            tide_elev=tide.tide_elev[
                tide.pos - tide.index_offset : tide_end - tide.index_offset
            ],
            land_elev=tide.land_elev[
                tide.pos - tide.index_offset : tide_end - tide.index_offset
            ],
            conc=np.zeros(
                len(
                    tide.index[
                        tide.pos - tide.index_offset : tide_end - tide.index_offset
                    ]
                )
            ),
        )
        self.deg_total = self.deg_total + (
            tide.land_elev[tide.pos - tide.index_offset]
            - tide.land_elev[tide_end - tide.index_offset]
        )

        self.df = self.df.append([tide_result, inundation_result])

    def plot(self, columns=None, frac=1.0):
        if columns is None:
            self.df.sample(frac=frac).sort_index().plot()
        else:
            self.df[columns].sample(frac=frac).sort_index().plot()


class Simulation:
    def __init__(
        self,
        tide_ts,
        land_elev_init,
        conc_init,
        conc_bound,
        grain_diam,
        grain_dens,
        bulk_dens,
        org_rate,
        comp_rate,
        sub_rate,
        slr,
        solver_method="RK45",
        years=1,
        min_depth=0.0,
        dense_output=False,
    ):
        self.params = Params()
        self.results = SimulationResults()

        assert isinstance(
            tide_ts, (pd.Series, pd.DataFrame)
        ), "tide_ts must be a Pandas Series or DataFrame."

        self.params.timestep = int(pd.to_timedelta(tide_ts.index.freq).total_seconds())
        vals = tide_ts.values
        tide_ts = pd.Series(
            data=vals,
            index=pd.RangeIndex(start=0, stop=len(vals), step=self.params.timestep),
        )

        self.params.conc_init = conc_init
        self.params.conc_bound = conc_bound
        self.params.grain_diam = grain_diam
        self.params.grain_dens = grain_dens
        self.params.bulk_dens = bulk_dens
        self.params.org_rate = org_rate
        self.params.comp_rate = comp_rate
        self.params.sub_rate = sub_rate
        self.params.slr = slr
        self.params.solver_method = solver_method
        self.params.min_depth = min_depth
        self.params.dense_output = dense_output
        self.tides = Tides(
            tide_ts=tide_ts, land_elev_init=land_elev_init, params=self.params
        )
        self.year = 0
        self.years = years
        self.start_time = None
        self.end_time = None
        self.pbar = None

    def __repr__(self):
        return "{name} @{id:x} {attrs}".format(
            name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    def step(self):
        inundation = self.tides.find_inundation()
        if inundation:
            inundation.aggrade()
            self.pbar.update(n=1.0)
        self.results.update(tide=self.tides, inundation=inundation)
        self.tides.update(inundation=inundation)

    def run(self, steps):
        while self.year < self.years:
            while self.tides.pos < self.tides.index[-1]:
                self.step()
                if self.pbar.n == self.pbar.total:
                    self.step()
                    return
                if steps is None:
                    self.pbar.total = (
                        self.pbar.n + self.tides.remaining_inundations
                    ) * (self.years - self.year)
            self.year += 1
            offset = self.tides.index[-1] + self.params.timestep
            tide_ts = pd.Series(
                data=self.tides.tide_elev + self.params.slr,
                index=self.tides.index + offset,
            )
            self.tides = Tides(
                tide_ts=tide_ts,
                land_elev_init=self.results.df.land_elev.iat[-1],
                params=self.params,
            )

    def _initialize(self, steps=None):
        self.start_time = time.perf_counter()
        self.pbar = tqdm(leave=True, position=0, unit="inundation", desc="Progress",)
        if steps:
            self.pbar.total = float(steps)
        else:
            self.pbar.total = self.tides.remaining_inundations * self.years

    def _uninitialize(self):
        self.end_time = time.perf_counter()
        self.pbar.close()
        self.print_results()

    def simulate(self, steps=None):
        self._initialize(steps=steps)
        self.run(steps=steps)
        self._uninitialize()

    @property
    def runtime(self):
        return time.strftime("%H:%M:%S", time.gmtime(self.end_time - self.start_time))

    def print_results(self):
        print("-" * 40)
        print(
            "{:<25} {:< 10.3f} {:>2}".format(
                "Starting elevation: ", self.results.df.land_elev.iat[0], "m"
            )
        )
        print(
            "{:<25} {:< 10.3f} {:>2}".format(
                "Final elevation: ", self.results.df.land_elev.iat[-1], "m"
            )
        )
        print(
            "{:<25} {:< 10.3f} {:>2}".format(
                "Elevation change: ", self.results.land_elev_change * 100, "cm"
            )
        )
        print("-" * 40)
        print(
            "{:<25} {:< 10.3f} {:>2}".format(
                "Aggradation: ", self.results.agg_total * 100, "cm"
            )
        )
        print(
            "{:<25} {:<10.3f} {:>2}".format(
                "Degradation: ", self.results.deg_total * 100, "cm"
            )
        )
        print("-" * 40)
        print("{:<25} {:>12}".format("Runtime: ", self.runtime))

