import numpy as np
import pandas as pd
import seaborn as sns
import time

from collections import namedtuple
from matplotlib import pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

sns.set()


def apply_rate(init_val, index, timestep_sec, rate_per_sec):
    weights = np.arange(0, len(index), 1) * timestep_sec
    vals = init_val + (weights * rate_per_sec)
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


class Params:
    def __init__(
        self,
        timestep,
        conc_init,
        conc_bound,
        grain_diam,
        grain_dens,
        bulk_dens,
        org_rate,
        comp_rate,
        sub_rate,
        slr,
        method,
        min_depth,
        dense_output,
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
        super(OdeResult, self).__init__(result)
        self.aggradation = self.y[2][-1] - self.y[2][0]
        self.degradation = None

    def __repr__(self):
        return "{name} @{id:x} {attrs}".format(
            name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    def plot(self):
        fig, axs = plt.subplots(3, 1)
        fig.set_figheight(10)
        fig.set_figwidth(15)

        # plot concentration
        time = (self.t - self.t[0]) / 60
        df = pd.DataFrame(
            data={
                "tide_elev": self.y[0],
                "conc": self.y[1],
                "land_elev_change": (self.y[2] - self.y[2][0]) * 1000,
            },
            index=time,
        )
        sns.lineplot(ax=axs[0], data=df["tide_elev"], color="b")
        axs[0].set(ylabel="Tide Elevation (m)")
        axs[0].set(xticklabels=[])

        sns.lineplot(ax=axs[1], data=df["conc"], color="r")
        axs[1].set(ylabel="Concentration (g/L)",)
        axs[1].set(xticklabels=[])

        sns.lineplot(ax=axs[2], data=df["land_elev_change"], color="g")
        axs[2].set(ylabel="Land Elevation $\Delta$ (mm)", xlabel="Time (min)")


class InundationResult:
    def __init__(self,):
        self.flood = None
        self.ebb = None

    def __repr__(self):
        return "{name} @{id:x} {attrs}".format(
            name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    @property
    def time(self):
        return np.append(self.flood.t, self.ebb.t[1:])

    @property
    def tide_elev(self):
        return np.append(self.flood.y[0], self.ebb.y[0][1:])

    @property
    def slack_time(self):
        return self.ebb.t[0]

    @property
    def conc(self):
        return np.append(self.flood.y[1], self.ebb.y[1][1:])

    @property
    def land_elev(self):
        return np.append(self.flood.y[2], self.ebb.y[2][1:])

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

        # plot concentration
        time = (self.time - self.time[0]) / 60
        df = pd.DataFrame(
            data={
                "tide_elev": self.tide_elev,
                "conc": self.conc,
                "land_elev_change": (self.land_elev - self.land_elev[0]) * 1000,
            },
            index=time,
        )
        sns.lineplot(ax=axs[0], data=df["tide_elev"], color="b")
        axs[0].set(ylabel="Tide Elevation (m)")
        axs[0].set(xticklabels=[])

        sns.lineplot(ax=axs[1], data=df["conc"], color="r")
        axs[1].set(ylabel="Concentration (g/L)",)
        axs[1].set(xticklabels=[])

        sns.lineplot(ax=axs[2], data=df["land_elev_change"], color="g")
        axs[2].set(ylabel="Land Elevation $\Delta$ (mm)", xlabel="Time (min)")

        for ax in axs:
            ax.axvline((self.slack_time - self.time[0]) / 60, color="k", linestyle="--")


class Inundation:
    def __init__(
        self, index, tide_elev, land_elev_init,
    ):
        self.index = index
        self.tide_elev = tide_elev
        self.land_elev_init = land_elev_init
        self.params = None
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

        flood = solve_ivp(
            fun=aggrade_flood,
            t_span=[self.pos_start, self.pos_slack],
            y0=[self.tide_elev[0], self.params.conc_init, self.land_elev_init],
            method=self.params.method,
            dense_output=self.params.dense_output,
            args=self.solver_args,
        )
        self.result.flood = LimbResult(flood)
        self.result.flood.degradation = (
            self.pos_slack - self.pos_start
        ) * self.params.linear_rate_sec

        assert flood.success is True, "[t={}] Flood integration failed!".format(
            self.index[0]
        )
        assert (
            flood.y[0] >= flood.y[2]
        ).all(), "[t={}] Depth cannot be negative!\ndepths={}".format(
            self.index[0], self.tide_elev_func(flood.t) - flood.y[1]
        )
        assert (
            flood.y[1][:-1] >= 0
        ).all(), "[t={}] Concentration cannot be negative!\nconc={}".format(
            self.index[0], flood.y[1]
        )
        # assert (self.slack_elev - self.land_elev_init) * self.conc_bound >= (flood.y[2][-1] - flood.y[2][0]) * self.bulk_dens, "Too much mass extracted! Available: {:.2e}, Extracted: {:.2e}".format((self.slack_elev - self.land_elev_init) * self.conc_bound, (flood.y[2][-1] - flood.y[2][0]) * self.bulk_dens)

        ebb = solve_ivp(
            fun=aggrade_ebb,
            t_span=[self.pos_slack, self.pos_end],
            y0=[self.tide_elev_slack, flood.y[1][-1], flood.y[2][-1]],
            events=(zero_conc, zero_depth),
            method=self.params.method,
            dense_output=self.params.dense_output,
            args=self.solver_args,
        )
        self.result.ebb = LimbResult(ebb)
        self.result.ebb.degradation = (
            self.pos_end - self.pos_slack
        ) * self.params.linear_rate_sec

        assert ebb.success is True, "[t={}] Ebb integration failed!".format(ebb.t[0])
        assert (
            ebb.y[0] >= ebb.y[2]
        ).all(), "[t={}] Depth cannot be negative!\ndepths={}".format(
            ebb.t[0], ebb.y[0] - ebb.y[1]
        )
        assert (
            ebb.y[1][:-1] >= 0
        ).all(), "[t={}] Concentration cannot be negative!\nconc={}".format(
            ebb.t[0], ebb.y[1]
        )
        # assert ebb.y[1][0] >= (ebb.y[2][-1] - ebb.y[2][0]) * self.bulk_dens, "Too much mass extracted! Available: {:.2e}, Extracted: {:.2e}".format(ebb.y[1][0], (ebb.y[2][-1] - ebb.y[2][0]) * self.bulk_dens)


class Tides:
    def __init__(self, index, tide_elev, land_elev, min_depth):
        try:
            assert len(tide_elev) == len(land_elev) == len(index)
        except AssertionError:
            print("Length of input arrays do not match.")

        assert (
            tide_elev[0] <= land_elev[0]
        ), "Tide elevation must start below the land elevation."

        self.index = index
        self.tide_elev = tide_elev
        self.land_elev = land_elev
        self.min_depth = min_depth
        self.timestep = index[1] - index[0]

    def __repr__(self):
        return "{name} @{id:x} {attrs}".format(
            name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    @property
    def depth(self):
        return self.tide_elev - self.land_elev

    @property
    def zero_crossings(self):
        crossings = (
            np.where(
                np.diff(np.signbit(self.tide_elev - (self.land_elev + self.min_depth)))
            )[0]
            + 1
        )
        return crossings

    def find_inundation(self, pos):
        remaining_crossings = self.zero_crossings[self.zero_crossings > pos]
        while True:
            if len(remaining_crossings) == 0:
                pos = self.index[-1]
                return None
            elif len(remaining_crossings) == 1:
                print("Warning: Partial inundation cycle!")
                start = remaining_crossings[0]
                end = self.index[-1]
            else:
                start, end = remaining_crossings[0:2]

            if not (self.tide_elev[start:end] > self.land_elev[start:end]).all():
                raise Exception

            if end - start < 4:
                pos = end + self.timestep
                print("Small inundation at t={}->{}. Skipping.")
                continue

            pos = start

            inundation = Inundation(
                index=self.index[start:end],
                tide_elev=self.tide_elev[start:end],
                land_elev_init=self.land_elev[start],
            )

            return inundation

    def plot(self):
        plt.plot(self.index, self.tide_elev, color="b", alpha=0.65)
        plt.plot(self.index, self.land_elev, color="g")


class SimulationResults:
    def __init__(self):
        self.data = pd.DataFrame(columns=["tide_elev", "land_elev"])
        self.degradation = 0.0
        self.aggradation = 0.0
        self.start_time = time.perf_counter()
        self.end_time = None

    def __repr__(self):
        return "{name} @{id:x} {attrs}".format(
            name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    @property
    def runtime(self):
        return self.end_time - self.start_time

    @property
    def land_elev_change(self):
        return self.data.iloc[-1].land_elev - self.data.iloc[0].land_elev

    def append(self, index, tide_elev, land_elev):
        new_data = pd.DataFrame(
            data={"tide_elev": tide_elev, "land_elev": land_elev}, index=index
        )
        self.data = self.data.append(new_data)

    def plot(self, frac=1.0):
        self.data.sample(frac=frac).sort_index().plot()


class Simulation:
    def __init__(
        self,
        tide_elev,
        land_elev_init,
        index,
        conc_init,
        conc_bound,
        grain_diam,
        grain_dens,
        bulk_dens,
        org_rate,
        comp_rate,
        sub_rate,
        slr,
        method="RK45",
        runs=1,
        min_depth=0.0,
        dense_output=False,
        save_inundations=False,
    ):
        self.params = Params(
            timestep=index[1] - index[0],
            conc_init=conc_init,
            conc_bound=conc_bound,
            grain_diam=grain_diam,
            grain_dens=grain_dens,
            bulk_dens=bulk_dens,
            org_rate=org_rate,
            comp_rate=comp_rate,
            sub_rate=sub_rate,
            slr=slr,
            method=method,
            min_depth=min_depth,
            dense_output=dense_output,
        )
        self.tides = Tides(
            index=index,
            tide_elev=tide_elev,
            land_elev=apply_rate(
                init_val=land_elev_init,
                index=index,
                timestep_sec=self.params.timestep,
                rate_per_sec=self.params.linear_rate_sec,
            ),
            min_depth=min_depth,
        )
        self.runs = runs
        self.pos = 0
        self.results = SimulationResults()
        self.inundations = []
        self.status = 0
        self.save_inundations = save_inundations

    def __repr__(self):
        return "{name} @{id:x} {attrs}".format(
            name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs="".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )

    def update_land(self, inundation: Inundation):
        before = self.tides.land_elev[: inundation.pos_end]
        after = apply_rate(
            init_val=inundation.result.land_elev[-1],
            index=self.tides.index[inundation.pos_end :],
            timestep_sec=inundation.params.timestep,
            rate_per_sec=inundation.params.linear_rate_sec,
        )
        new = np.append(before, after)

        assert len(new) == len(
            self.tides.land_elev
        ), "Length of new land elevation array does not match old array."

        self.tides.land_elev = new

    def update_results(self, inundation: Inundation):
        self.results.append(
            index=self.tides.index[self.pos : inundation.pos_start],
            tide_elev=self.tides.tide_elev[self.pos : inundation.pos_start],
            land_elev=self.tides.land_elev[self.pos : inundation.pos_start],
        )
        self.results.degradation = (
            self.results.degradation
            + self.params.linear_rate_sec * (inundation.pos_start - self.pos)
        )
        self.results.append(
            index=inundation.result.time,
            tide_elev=inundation.result.tide_elev,
            land_elev=inundation.result.land_elev,
        )
        self.results.aggradation = (
            self.results.aggradation + inundation.result.aggradation
        )
        self.results.degradation = (
            self.results.degradation + inundation.result.degradation
        )

    def update(self, inundation=None):
        if inundation:
            if self.save_inundations is True:
                self.inundations.append(inundation)
            self.update_results(inundation)
            self.update_land(inundation)
            self.pos = inundation.index[-1] + self.params.timestep
        elif inundation is None:
            self.results.append(
                index=self.tides.index[self.pos :],
                tide_elev=self.tides.tide_elev[self.pos :],
                land_elev=self.tides.land_elev[self.pos :],
            )
            self.pos = self.tides.index[-1]
            self.results.end_time = time.perf_counter()

    def simulate(self):
        while self.status == 0:
            inundation = self.tides.find_inundation(pos=self.pos)
            if inundation:
                inundation.params = self.params
                inundation.aggrade()
            else:
                self.status = 1

            self.update(inundation=inundation)

        self.print_results()

    def print_results(self):
        print("Aggradation:           {:.2e} m".format((self.results.aggradation)))
        print("Degradation:           {:.2e} m".format(self.results.degradation))
        print("Elevation change:      {:.2e} m".format(self.results.land_elev_change))
        print(
            "Final elevation:       {:.4f} m".format(
                self.results.data.iloc[-1].land_elev
            )
        )
        print(
            "Runtime:               {}".format(
                time.strftime("%H:%M:%S", time.gmtime(self.results.runtime))
            )
        )

    # def run_model(self, runs):
    #     for run in range(0, self.runs):
    #         offset = (len(self.tide_elev) * self.timestep) * run
    #         self.tides.index = self.tides.index + offset
    #         self.simulate(tides = self.tides)

    #         self.land_elev_init = self.results[-1].land_elev[-1]
    #         self.tides.tide_elev = self.tide_elev + self.slr
    #         self.tides.land_elev = self.land_elev
