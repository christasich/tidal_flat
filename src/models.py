import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
import scipy
import time
import logging
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

logger = logging.getLogger(__name__)

def apply_rate(init_val, index, timestep_sec, rate_per_sec):
    weights = np.arange(0, len(index), 1) * timestep_sec
    vals = init_val + (weights * rate_per_sec)
    return vals

def stokes_settling(grain_dia, grain_dens, fluid_dens=1000.0, fluid_visc=0.001, g=9.8):
    """
    Function that uses Stokes' Law to calculate the settling velocity of a spherical particle
    falling through a fluid column.
    """
    settle_rate = (
        (2 / 9 * (grain_dens - fluid_dens) / fluid_visc) * g * (grain_dia / 2) ** 2
    )
    return settle_rate

def aggrade(t, y, limb, dZdt_func, bound_conc, settle_rate, bulk_dens):

    # set values for concentration and elevation
    tide_elev = y[0]
    conc = y[1]
    land_elev = y[2]

    # use spline function for tide height to set current water_height
    depth = tide_elev - land_elev  # calculate current depth

    # use derivative of tide spline to get current gradient and set H
    dZdt = dZdt_func(t)

    if limb=='flood':
        dCdt = - (settle_rate * conc) / depth - 1 / depth * (conc - bound_conc) * dZdt
    elif limb=='ebb':
        dCdt = - (settle_rate * conc) / depth

    dEdt = settle_rate * conc / bulk_dens

    return [dZdt, dCdt, dEdt]

def zero_conc(t, y, *args):
    return y[1]

zero_conc.terminal = True
zero_conc.direction = -1

def zero_depth(t, y, *args):
    return y[0] + 0.001

zero_depth.terminal = True
zero_depth.direction = -1




class InundationODE(OdeResult):

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
                "land_elev_delta": (self.y[2] - self.y[2][0]) * 1000
            },
            index=time)
        sns.lineplot(ax=axs[0], data=df[ "tide_elev"], color="b")
        axs[0].set(ylabel="Tide Elevation (m)")
        axs[0].set(xticklabels=[])

        sns.lineplot(ax=axs[1], data=df[ "conc"], color="r")
        axs[1].set(
            ylabel="Concentration (g/L)",
            )
        axs[1].set(xticklabels=[])

        sns.lineplot(ax=axs[2], data=df[ "land_elev_delta"], color="g")
        axs[2].set(
            ylabel="Land Elevation $\Delta$ (mm)",
            xlabel="Time (min)"
            )

class InundationResult:
    def __init__(
        self,
        flood=None,
        ebb=None,
    ):
        self.flood = flood
        self.ebb = ebb

    def __repr__(self):
        return "<{} @{id:x} {attrs}>".format(
            self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=" ".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
            )

    @property
    def time(self):
        return np.append(self.flood.t, self.ebb.t[1:])

    @property
    def tide_elev(self):
        return np.append(self.flood.y[0], self.ebb.y[0][1:])

    @property
    def conc(self):
        return np.append(self.flood.y[1], self.ebb.y[1][1:])

    @property
    def land_elev(self):
        return np.append(self.flood.y[2], self.ebb.y[2][1:])

    @property
    def land_elev_start(self):
        return self.land_elev[0]

    @property
    def land_elev_end(self):
        return self.land_elev[-1]

    @property
    def land_elev_delta(self):
        return self.land_elev[-1] - self.land_elev[0]

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
                "land_elev_delta": (self.land_elev - self.land_elev_start) * 1000
            },
            index=time)
        sns.lineplot(ax=axs[0], data=df[ "tide_elev"], color="b")
        axs[0].set(ylabel="Tide Elevation (m)")
        axs[0].set(xticklabels=[])

        sns.lineplot(ax=axs[1], data=df[ "conc"], color="r")
        axs[1].set(
            ylabel="Concentration (g/L)",
            )
        axs[1].set(xticklabels=[])

        sns.lineplot(ax=axs[2], data=df[ "land_elev_delta"], color="g")
        axs[2].set(
            ylabel="Land Elevation $\Delta$ (mm)",
            xlabel="Time (min)"
            )

        for ax in axs:
            ax.axvline((self.ebb.t[0] - self.time[0]) / 60, color="k", linestyle="--")


class Inundation:
    def __init__(
        self,
        tide_elev,
        init_land_elev,
        index,
        pos,
        init_conc=None,
        bound_conc=None,
        settle_rate=None,
        bulk_dens=None,
        result=None,
    ):
        self.tide_elev = tide_elev
        self.init_land_elev = init_land_elev
        self.index = index
        self.pos =  pos
        self.init_conc = init_conc
        self.bound_conc = bound_conc
        self.settle_rate = settle_rate
        self.bulk_dens = bulk_dens
        self.result = result

    def __repr__(self):
        return "<{} @{id:x} {attrs}>".format(
            self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=" ".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
            )

    @property
    def tide_spline(self):
        return InterpolatedUnivariateSpline(x=self.index, y=self.tide_elev)

    @property
    def tide_spline_deriv(self):
        return self.tide_spline.derivative()

    @property
    def slack_time(self):
        return np.argmax(self.tide_elev) + self.pos

    @property
    def slack_elev(self):
        return np.max(self.tide_elev)

    @property
    def start(self):
        return self.index[0]

    @property
    def end(self):
        return self.index[-1]

    @property
    def period(self):
        return self.index[-1] - self.index[0]


    def aggrade(self):
        self.result = InundationResult()

        flood = solve_ivp(
            fun=aggrade,
            t_span=[self.index[0], self.slack_time],
            y0=[self.tide_elev[0], self.init_conc, self.init_land_elev],
            method="DOP853",
            args=(
                "flood",
                self.tide_spline_deriv,
                self.bound_conc,
                self.settle_rate,
                self.bulk_dens,
                )
            )
        self.result.flood = InundationODE(flood)
        assert flood.success is True, "[t={}] Flood integration failed!".format(self.index[0])
        assert (self.slack_elev - self.init_land_elev) * self.bound_conc >= (flood.y[2][-1] - flood.y[2][0]) * self.bulk_dens, "Too much mass extracted! Available: {:.2e}, Extracted: {:.2e}".format((self.slack_elev - self.init_land_elev) * self.bound_conc, (flood.y[2][-1] - flood.y[2][0]) * self.bulk_dens)
        assert (flood.y[0] >= flood.y[2]).all(), "[t={}] Depth cannot be negative!\ndepths={}".format(self.index[0], self.tide_spline(flood.t) - flood.y[1])
        assert (flood.y[1][:-1] >= 0).all(), "[t={}] Concentration cannot be negative!\nconc={}".format(self.index[0], flood.y[1])

        ebb = solve_ivp(
            fun=aggrade,
            t_span=[self.slack_time, self.index[-1]],
            y0=[self.slack_elev, flood.y[1][-1], flood.y[2][-1]],
            events=(zero_conc, zero_depth),
            method="DOP853",
            args=(
                "ebb",
                self.tide_spline_deriv,
                self.bound_conc,
                self.settle_rate,
                self.bulk_dens,
                )
            )
        self.result.ebb = InundationODE(ebb)
        assert ebb.success is True, "[t={}] Ebb integration failed!".format(ebb.t[0])
        assert ebb.y[1][0] >= (ebb.y[2][-1] - ebb.y[2][0]) * self.bulk_dens, "Too much mass extracted! Available: {:.2e}, Extracted: {:.2e}".format(ebb.y[1][0], (ebb.y[2][-1] - ebb.y[2][0]) * self.bulk_dens)
        assert (ebb.y[0] >= ebb.y[2]).all(), "[t={}] Depth cannot be negative!\ndepths={}".format(ebb.t[0], ebb.y[0] - ebb.y[1])
        assert (ebb.y[1][:-1] >= 0).all(), "[t={}] Concentration cannot be negative!\nconc={}".format(ebb.t[0], ebb.y[1])

class SimulationResults:
    def __init__(self):
        self.data = pd.DataFrame(columns=["tide_elev", "land_elev"])
        self.degradation = 0.0
        self.aggradation = 0.0
        self.start_time = time.perf_counter()
        self.end_time = None

    @property
    def run_time(self):
        return self.end_time - self.start_time

    @property
    def land_elev_delta(self):
        return self.data.iloc[-1].land_elev - self.data.iloc[0].land_elev

    def append(self, index, tide_elev, land_elev):
        new_data = pd.DataFrame(
            data={
                "tide_elev": tide_elev,
                "land_elev": land_elev
                },
            index=index
        )
        self.data = self.data.append(new_data)


class Params:
    def __init__(
        self,
        init_land_elev,
        init_conc,
        bound_conc,
        grain_dia,
        grain_dens,
        bulk_dens,
        org_rate,
        comp_rate,
        sub_rate,
    ):
        self.init_land_elev = init_land_elev
        self.init_conc = init_conc
        self.bound_counc = bound_conc
        self.grain_dia = grain_dia
        self.grain_dens = grain_dens
        self.bulk_dens = bulk_dens
        self.org_rate = org_rate
        self.comp_rate = comp_rate
        self.sub_rate = sub_rate

    @property
    def settle_rate(self):
        return stokes_settling(
                grain_dia=self.grain_dia,
                grain_dens=self.grain_dens,
                fluid_dens=1000.0,
                fluid_visc=0.001,
                g=9.8)

    @property
    def linear_rate_sec(self):
        return (self.org_rate - self.comp_rate - self.sub_rate) / 365 / 24 / 60 / 60

class Tides:
    def __init__(self, index, tide_elev, land_elev):
        try:
            assert len(tide_elev) == len(land_elev) == len(index)
        except AssertionError:
            print("Length of input arrays do not match.")

        assert tide_elev[0] <= land_elev[0], "Tide elevation must start below the land elevation."

        self.tide_elev = tide_elev
        self.land_elev = land_elev
        self.index = index
        self.timestep = index[1] - index[0]

    def __repr__(self):
        return "<{} @{id:x} {attrs}>".format(
            self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=" ".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
            )

    @property
    def depth(self):
        return self.tide_elev - self.land_elev

    @property
    def zero_crossings(self):
        crossings = np.where(np.diff(np.signbit(self.tide_elev - (self.land_elev + 0.001))))[0] + 1
        return crossings

    def update(self, new_land_elev, pos, linear_rate=0.0):
        before = self.land_elev[:pos]
        after = apply_rate(init_val=new_land_elev, index=self.index[pos:], timestep_sec=self.timestep, rate_per_sec=linear_rate)
        new = np.append(before, after)

        assert len(new) == len(self.land_elev), "Length of new land elevation array does not match old array."

        self.land_elev = new

    def find_inundation(self, pos):
        remaining_crossings = self.zero_crossings[self.zero_crossings > pos]
        while True:
            if len(remaining_crossings) == 0:
                pos = self.index[-1]
                return None
            elif len(remaining_crossings) == 1:
                print('Warning: Partial inundation cycle!')
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
                tide_elev=self.tide_elev[start:end],
                init_land_elev=self.land_elev[start],
                index=self.index[start:end],
                pos=pos,
                )

            return inundation

class Simulation:
    def __init__(
        self,
        tide_elev,
        init_land_elev,
        index,
        init_conc,
        bound_conc,
        grain_dia,
        grain_dens,
        bulk_dens,
        org_rate,
        comp_rate,
        sub_rate,
        slr,
        pos=0,
        runs=1,
        inundations=[],
    ):
        self.params = Params(
            init_land_elev=self.init_land_elev,
            init_conc=self.init_conc,
            bound_conc=self.bound_conc,
            grain_dia=self.grain_dia,
            grain_dens=self.grain_dens,
            bulk_dens=self.bulk_dens,
            org_rate=self.org_rate,
            comp_rate=self.comp_rate,
            sub_rate=self.sub_rate,
            )
        self.tide_elev = tide_elev
        self.init_land_elev = init_land_elev
        self.index = index
        self.timestep = index[1] - index[0]
        self.init_conc = init_conc
        self.conc = np.zeros(len(index))
        self.bound_conc = bound_conc
        self.grain_dia = grain_dia
        self.grain_dens = grain_dens
        self.bulk_dens = bulk_dens
        self.org_rate = org_rate
        self.comp_rate = comp_rate
        self.sub_rate = sub_rate
        self.linear_rate_sec = (self.org_rate - self.comp_rate - self.sub_rate) / 365 / 24 / 60 / 60
        self.land_elev = apply_rate(
            init_val=self.init_land_elev,
            index=self.index,
            timestep_sec=self.timestep,
            rate_per_sec=self.linear_rate_sec
            )
        self.slr = slr
        self.pos =  pos
        self.runs = runs
        self.inundations = inundations
        self.results = SimulationResults()

    def __repr__(self):
        return "<{} @{id:x} {attrs}>".format(
            self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=" ".join("\n{}={!r}".format(k, v) for k, v in self.__dict__.items()),
            )

    @property
    def settle_rate(self):
        return stokes_settling(
                grain_dia=self.grain_dia,
                grain_dens=self.grain_dens,
                fluid_dens=1000.0,
                fluid_visc=0.001,
                g=9.8)

    @property
    def depth(self):
        return self.tide_elev - self.land_elev

    @property
    def zero_crossings(self):
        crossings = np.where(np.diff(np.signbit(self.tide_elev - (self.land_elev + 0.001))))[0] + 1
        return crossings

    def find_inundation(self):
        pos = self.pos
        remaining_crossings = self.zero_crossings[self.zero_crossings > pos]
        while True:
            if len(remaining_crossings) == 0:
                pos = self.index[-1]
                return None
            elif len(remaining_crossings) == 1:
                print('Warning: Partial inundation cycle!')
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
                tide_elev=self.tide_elev[start:end],
                init_land_elev=self.land_elev[start],
                index=self.index[start:end],
                pos=pos,
                bound_conc=self.bound_conc,
                settle_rate=self.settle_rate,
                init_conc=self.init_conc,
                bulk_dens=self.bulk_dens,
                )

            return inundation

    def update_land(self, inundation):
        before = self.land_elev[:inundation.end]
        after = apply_rate(
            init_val=inundation.result.land_elev_end - self.linear_rate_sec * inundation.period,
            index=self.index[inundation.end:],
            timestep_sec=self.timestep,
            rate_per_sec=self.linear_rate_sec
            )
        new = np.append(before, after)

        assert len(new) == len(self.land_elev), "Length of new land elevation array does not match old array."

        self.land_elev = new

    def update_results(self, inundation=None, degradation=None):
        if inundation and degradation:
            raise Exception("Can only update inundation or degradation. Not both!")
        elif inundation:
            self.results.append(
                index=inundation.result.time,
                tide_elev=inundation.result.tide_elev,
                land_elev=inundation.result.land_elev - self.linear_rate_sec * inundation.period
                )
            self.results.aggradation = self.results.aggradation + inundation.result.land_elev_delta
            self.results.degradation = self.results.degradation + self.linear_rate_sec * inundation.period
        elif degradation:
            self.results.degradation = self.results.degradation + degradation


    def simulate(self):
        inundation = self.find_inundation()
        while inundation is not None:
            self.results.append(
                index=self.index[self.pos:inundation.start],
                tide_elev=self.tide_elev[self.pos:inundation.start],
                land_elev=self.land_elev[self.pos:inundation.start],
                )
            self.inundations.append(inundation)
            self.update_results(degradation=self.linear_rate_sec * (inundation.start - self.pos))
            inundation.aggrade()
            self.update_results(inundation=inundation)
            self.update_land(inundation=inundation)
            self.pos = inundation.index[-1] + self.timestep
            inundation = self.find_inundation()

        self.results.append(
            index=self.index[self.pos:],
            tide_elev=self.tide_elev[self.pos:],
            land_elev=self.land_elev[self.pos:],
            )
        self.results.end_time = time.perf_counter()
        self.pos = self.index[-1]
        self.print_results()

    def print_results(self):
        print("Aggradation:           {:.2e} m".format((self.results.aggradation)))
        print("Degradation:           {:.2e} m".format(self.results.degradation))
        print("Elevation change:      {:.2e} m".format(self.results.land_elev_delta))
        print("Final elevation:       {:.4f} m".format(self.results.data.iloc[-1].land_elev))
        print("Runtime:               {}".format(time.strftime("%H:%M:%S", time.gmtime(self.results.run_time))))


    def run_model(self, runs):
        for run in range(0, self.runs):
            offset = (len(self.tide_elev) * self.timestep) * run
            self.tides.index = self.tides.index + offset
            self.simulate(tides = self.tides)

            self.init_land_elev = self.results[-1].land_elev[-1]
            self.tides.tide_elev = self.tide_elev + self.slr
            self.tides.land_elev = self.land_elev