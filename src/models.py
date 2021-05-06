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

    # set values for concentration and elevation
    tide_elev = y[0]
    conc = y[1]
    land_elev = y[2]

    # use spline function for tide height to set current water_height
    depth = tide_elev - land_elev  # calculate current depth

    # use derivative of tide spline to get current gradient and set H
    dZdt = args.dZdt_func(t)
    dCdt = (
        -(args.settle_rate * conc) / depth - 1 / depth * (conc - args.conc_bound) * dZdt
    )
    dEdt = args.settle_rate * conc / args.bulk_dens

    return [dZdt, dCdt, dEdt]


def aggrade_ebb(
    t, y, args,
):

    # set values for concentration and elevation
    tide_elev = y[0]
    conc = y[1]
    land_elev = y[2]

    # use spline function for tide height to set current water_height
    depth = tide_elev - land_elev  # calculate current depth

    # use derivative of tide spline to get current gradient and set H
    dZdt = args.dZdt_func(t)
    dCdt = -(args.settle_rate * conc) / depth
    dEdt = args.settle_rate * conc / args.bulk_dens

    return [dZdt, dCdt, dEdt]


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
        return (self.org_rate - self.comp_rate - self.sub_rate) / 365 / 24 / 60 / 60


class LimbResult(OdeResult):
    def __init__(self, result):
        super().__init__(result)
        self.aggradation = self.y[2][-1] - self.y[2][0]
        self.degradation = None

    def __repr__(self):
        return "{name} @{id:x} {attrs}".format(
            name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
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
    def df(self):
        return tide_df(
            index=self.time,
            tide_elev=self.tide_elev,
            land_elev=self.land_elev,
            conc=self.conc,
        )

    def plot(self):
        fig, axs = plt.subplots(3, 1)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        time = self.time / 60

        sns.lineplot(ax=axs[0], x=time, y=self.tide_elev, color="b")
        axs[0].set(ylabel="Tide Elevation (m)")
        axs[0].set(xticklabels=[])

        sns.lineplot(ax=axs[1], x=time, y=self.conc, color="r")
        axs[1].set(ylabel="Concentration (g/L)",)
        axs[1].set(xticklabels=[])

        sns.lineplot(ax=axs[2], x=time, y=self.land_elev, color="g")
        axs[2].set(ylabel="Land Elevation $\Delta$ (mm)", xlabel="Time (min)")


class InundationResult:
    def __init__(self, flood=None, ebb=None):
        self.flood = flood
        self.ebb = ebb

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
    def df(self):
        return tide_df(
            index=self.time,
            tide_elev=self.tide_elev,
            land_elev=self.land_elev,
            conc=self.conc,
        )

    @property
    def slack_time(self):
        return self.ebb.t[0]

    @property
    def aggradation(self):
        return self.flood.aggradation + self.ebb.aggradation

    @property
    def degradation(self):
        return self.flood.degradation + self.ebb.degradation

    def plot(self):
        fig, axs = plt.subplots(3, 1)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        time = self.time / 60

        # plot concentration
        sns.lineplot(ax=axs[0], x=time, y=self.tide_elev, color="b")
        axs[0].set(ylabel="Tide Elevation (m)")
        axs[0].set(xticklabels=[])

        sns.lineplot(ax=axs[1], x=time, y=self.conc, color="r")
        axs[1].set(ylabel="Concentration (g/L)",)
        axs[1].set(xticklabels=[])

        sns.lineplot(ax=axs[2], x=time, y=self.land_elev, color="g")
        axs[2].set(ylabel="Land Elevation $\Delta$ (mm)", xlabel="Time (min)")

        for ax in axs:
            ax.axvline((self.slack_time), color="k", linestyle="--")


class Inundation:
    def __init__(
        self, index, tide_elev, land_elev_init, params,
    ):
        self.index = index
        self.tide_elev = tide_elev
        self.land_elev_init = land_elev_init
        self.params = params
        self.result = None

    def __repr__(self):
        return "{name} @{id:x} {attrs}".format(
            name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    @property
    def tide_elev_func(self):
        return InterpolatedUnivariateSpline(x=self.index, y=self.tide_elev)

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
        return self.index[-1] - self.index[0]

    @property
    def tide_elev_slack(self):
        return np.max(self.tide_elev)

    @property
    def solver_args(self):
        args = namedtuple(
            "aggrade_args",
            ["dZdt_func", "conc_bound", "settle_rate", "bulk_dens", "min_depth"],
        )
        return [
            args(
                dZdt_func=self.tide_elev_func.derivative(),
                conc_bound=self.params.conc_bound,
                settle_rate=self.params.settle_rate,
                bulk_dens=self.params.bulk_dens,
                min_depth=self.params.min_depth,
            )
        ]

    def aggrade(self):
        self.result = InundationResult()

        # Integrate flood limb
        flood = solve_ivp(
            fun=aggrade_flood,
            t_span=[self.pos_start, self.pos_slack],
            y0=[self.tide_elev[0], self.params.conc_init, self.land_elev_init],
            method=self.params.method,
            dense_output=self.params.dense_output,
            args=self.solver_args,
        )
        flood = LimbResult(flood)
        flood.degradation = (
            self.pos_slack - self.pos_start
        ) * self.params.linear_rate_sec

        assert flood.success is True, "[t={}] Flood integration failed!".format(
            self.pos_start
        )
        assert (
            flood.tide_elev >= flood.land_elev
        ).all(), "[t={}] Depth cannot be negative!\ndepths={}".format(
            self.pos_start, self.tide_elev_func(flood.time) - flood.land_elev
        )
        assert (
            flood.conc[:-1] >= 0
        ).all(), "[t={}] Concentration cannot be negative!\nconc={}".format(
            self.pos_start, flood.conc
        )
        self.result.flood = flood

        # Integrate ebb limb
        ebb = solve_ivp(
            fun=aggrade_ebb,
            t_span=[self.pos_slack, self.pos_end],
            y0=[self.tide_elev_slack, flood.y[1][-1], flood.y[2][-1]],
            events=(zero_conc, zero_depth),
            method=self.params.method,
            dense_output=self.params.dense_output,
            args=self.solver_args,
        )
        ebb = LimbResult(ebb)
        ebb.degradation = (self.pos_end - self.pos_slack) * self.params.linear_rate_sec

        assert ebb.success is True, "[t={}] Ebb integration failed!".format(
            self.pos_slack
        )
        assert (
            ebb.tide_elev >= ebb.land_elev
        ).all(), "[t={}] Depth cannot be negative!\ndepths={}".format(
            self.pos_slack, ebb.tide_elev - ebb.land_elev
        )
        assert (
            ebb.conc[:-1] >= 0
        ).all(), "[t={}] Concentration cannot be negative!\nconc={}".format(
            self.pos_slack, ebb.conc
        )

        self.result.ebb = ebb


class Tides:
    def __init__(self, tide_ts=None, land_elev_init=None, params=None):
        self.index = tide_ts.index.values
        self.tide_elev = tide_ts.values
        self.land_elev = land_elev_init - tide_ts.index.values * params.linear_rate_sec
        self.pos = 0
        self.params = params

    def __repr__(self):
        return "{name} @{id:x} {attrs}".format(
            name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    @property
    def df(self):
        return tide_df(
            index=self.index, tide_elev=self.tide_elev, land_elev=self.land_elev
        )

    def subset(self, start=0, end=None, include_conc=True):
        df = self.df.iloc[start:end]
        if include_conc is True:
            df["conc"] = 0
        return df

    def degradation(self, start=0, end=-1):
        return self.land_elev[start] - self.land_elev[end]

    @property
    def zero_crossings(self):
        crossings = (
            np.where(
                np.diff(
                    np.signbit(
                        self.tide_elev[self.pos :]
                        - (self.land_elev[self.pos :] + self.params.min_depth)
                    )
                )
            )[0]
            + 1
        )
        return crossings + self.pos

    def find_inundation(self, pbar=None):
        remaining_crossings = self.zero_crossings
        while True:
            if len(remaining_crossings) == 0:
                if pbar:
                    pbar.total = pbar.n + pbar.n * pbar.runs_left
                    pbar.refresh()
                return None
            elif len(remaining_crossings) == 1:
                print("Warning: Partial inundation cycle!")
                start = remaining_crossings[0]
                end = self.index[-1]
            elif len(remaining_crossings) > 1:
                start, end = remaining_crossings[0:2]

            if end - start < 4:
                remaining_crossings = remaining_crossings[2:]
                print("Small inundation at t={}->{}. Skipping.")
                continue

            subset = self.df.iloc[start:end]

            assert (subset.tide_elev > subset.land_elev + self.params.min_depth).all()

            inundation = Inundation(
                index=subset.index.values,
                tide_elev=subset.tide_elev.values,
                land_elev_init=subset.land_elev.values[0],
                params=self.params,
            )
            if pbar:
                pbar.total = (pbar.n + len(remaining_crossings) / 2) + (
                    pbar.n + len(remaining_crossings) / 2
                ) * pbar.runs_left
                pbar.update()

            return inundation

    def update(self, inundation: Inundation):
        if inundation is None:
            self.pos = self.index[-1]
            return
        before = self.land_elev[: inundation.pos_end]
        after = (
            inundation.result.land_elev[-1]
            - (self.index[inundation.pos_end :] - self.index[inundation.pos_end])
            * self.params.linear_rate_sec
        )
        new = np.append(before, after)

        assert len(new) == len(
            self.land_elev
        ), "Length of new land elevation ({:.2e}) array does match length of old array ({:.2e}). Diff = {:.2e}".format(
            len(new), len(self.land_elev), abs(len(new) - len(self.land_elev))
        )

        self.land_elev = new
        self.pos = inundation.pos_end + self.params.timestep


class SimulationResults:
    def __init__(self):
        self.inundations = []
        self.aggradation = 0.0
        self.degradation = 0.0
        self.start_time = time.perf_counter()
        self.end_time = None
        self.df = tide_df()

    def __repr__(self):
        return "{name} @{id:x} {attrs}".format(
            name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    @property
    def runtime(self):
        return self.end_time - self.end_time

    @property
    def land_elev_change(self):
        return self.df.land_elev.iat[-1] - self.df.land_elev.iat[0]

    def update(self, tide: Tides, inundation: Inundation):
        if inundation is not None:
            tide_df = tide.subset(start=tide.pos, end=inundation.pos_start)
            tide_degradation = tide_df.land_elev.iat[0] - tide_df.land_elev.iat[-1]
            self.df = self.df.append([tide_df, inundation.result.df])
            self.aggradation = self.aggradation + inundation.result.aggradation
            self.degradation = (
                self.degradation + tide_degradation + inundation.result.degradation
            )
            self.inundations.append(inundation)
        elif inundation is None:
            tide_df = tide.subset(start=tide.pos)
            tide_degradation = tide_df.land_elev.iat[0] - tide_df.land_elev.iat[-1]
            self.df = self.df.append(tide_df)
            self.degradation = self.degradation + tide_degradation

    def plot(self, frac=1.0):
        self.df.sample(frac=frac).sort_index().plot()


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
        runs=1,
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
        self.run = 0
        self.n = 0
        self.runs = runs
        self.pbar = tqdm(
            total=len(self.tides.zero_crossings) / 2 * self.runs, leave=True, position=0
        )
        self.pbar.runs_left = self.runs - 1.0

    def __repr__(self):
        return "{name} @{id:x} {attrs}".format(
            name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    def step(self, N=1):
        max_step = self.n + N
        while self.n < max_step:
            inundation = self.tides.find_inundation(pbar=self.pbar)
            try:
                inundation.aggrade()
            except AttributeError:
                pass

            self.results.update(tide=self.tides, inundation=inundation)
            self.tides.update(inundation=inundation)
            self.n += 1

    def run_one(self):
        while self.tides.pos < self.tides.index[-1]:
            self.step()
        self.run += 1
        self.pbar.runs_left = self.runs - self.run

    def run_all(self):
        while self.run < self.runs:
            self.run_one()
            offset = (len(self.tides.index) * self.params.timestep) * self.run
            tide_ts = self.tides.df.tide_elev + offset
            self.tides = Tides(
                tide_ts=tide_ts,
                land_elev_init=self.results.df.land_elev.iat[-1],
                params=self.params,
            )
        self.results.end_time = time.perf_counter()
        self.print_results()

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
                "Aggradation: ", self.results.aggradation * 100, "cm"
            )
        )
        print(
            "{:<25} {:<10.3f} {:>2}".format(
                "Degradation: ", self.results.degradation * 100, "cm"
            )
        )
        print("-" * 40)
        print(
            "{:<25} {:>12}".format(
                "Runtime: ",
                time.strftime("%H:%M:%S", time.gmtime(self.results.runtime)),
            )
        )

