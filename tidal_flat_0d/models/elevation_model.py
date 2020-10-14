import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import solve_ivp
import logging
logger = logging.getLogger(__name__)


def stokes_settling(grain_dia, grain_den, fluid_den = 1000, fluid_visc = 0.001, g = 9.8):
    settle_rate = (
        (2 / 9 * (grain_den - fluid_den) / fluid_visc) * g * (grain_dia / 2) ** 2
    )
    return settle_rate


def aggrade(water_heights, settle_rate, bulk_dens, bound_conc, init_elev=0, init_conc=0, timestep=1, depth_cutoff=0.001, debug=False):

    def below_platform(t, y, *args):
        depth = tide_spl(t) - (y[1] + depth_cutoff)
        return depth

    below_platform.terminal = True
    below_platform.direction = -1

    def solve_ode(t, y, *args):
        
        # set initial values for concentration and elevation
        init_conc = y[0]
        init_elev = y[1]
        
        # use spline function for tide height to set current water_height
        water_height = tide_spl(t)
        depth = water_height - init_elev #calculate current depth

        # use derivative of tide spline to get current gradient and set H
        dhdt = dhdt_spl(t)

        if dhdt > 0:
            H = 1
        else:
            H = 0
        
        delta_conc = - (settle_rate * init_conc) / depth - H / depth * (init_conc - bound_conc) * dhdt * timestep
        delta_elev = settle_rate * (init_conc + delta_conc) / bulk_dens * timestep

        return [delta_conc, delta_elev]

    elev = init_elev
    conc = init_conc
    data = pd.Series(water_heights)
    pos = 0
    elevs = np.empty(0)
    concs = np.empty(0)
    times = np.empty(0)
    inundations = 0

    while True:
        remaining_data = data[pos:]
        data_above_platform = remaining_data[remaining_data > (elev + depth_cutoff)]
        
        if len(data_above_platform) < 4:
            break

        if len(np.where(np.diff(data_above_platform.index.values) != 1)[0]) == 0:
            end = len(data_above_platform) - 1
        else:
            end = np.where(np.diff(data_above_platform.index.values) != 1)[0][0] + 1
        
        subset = data_above_platform[:end]

        if len(subset) < 4:
            pos = subset.index.values[-1] + 1
            continue

        subset_water_height = subset.values
        subset_index = subset.index.values

        tide_spl = InterpolatedUnivariateSpline(subset_index, subset_water_height)
        dhdt_spl = tide_spl.derivative()

        t_span = [subset_index[0], subset_index[-1]]
        result = solve_ivp(fun=solve_ode, t_span=t_span, y0=[conc, elev], events=below_platform,
                               args=(bound_conc, settle_rate, bulk_dens, depth_cutoff, timestep), dense_output=True)
        if result.status == -1:
            print(result.message)
            break
        
        times = np.concatenate((times, result.t))
        concs = np.concatenate((concs, result.y[0]))
        elevs = np.concatenate((elevs, result.y[1]))

        elev = result.y[1][-1]
        pos = subset.index.values[-1] + 1
        inundations += 1

    return [times, concs, elevs]