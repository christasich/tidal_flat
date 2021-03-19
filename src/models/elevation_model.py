import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import solve_ivp
import time
import logging
logger = logging.getLogger(__name__)


def stokes_settling(grain_dia, grain_den, fluid_den = 1000.0, fluid_visc = 0.001, g = 9.8):
    """
    Function that uses Stokes' Law to calculate the settling velocity of a spherical particle 
    falling through a fluid column.
    """
    settle_rate = ((2 / 9 * (grain_den - fluid_den) / fluid_visc) * g * (grain_dia / 2) ** 2)
    return settle_rate


def degrade_linear(init_elev, time_arr, comp_rate, sub_rate):
    """
    Function that linearly degrades land-surface elevation given an initial
    elevation, a time array of lenght N, compaction rate, and subsidence rate.
    Returns an array of elevations of length N.
    """
    comp_and_sub = time_arr * (comp_rate + sub_rate)
    elevs = init_elev - comp_and_sub

    return elevs


class ResultClass:
    """
    Result class to imitate scipy.integrate.solve_ivp result for consistency
    """

    def __init__(self, t=None, y=None):
        if t is None and y is None:
            self.t = np.empty(0)
            self.y = [np.empty(0), np.empty(0)]
        else:
            self.t = t
            self.y = y


def concatenate_results(results, new_result):
    """
    Function that concatenates results and returns as one ResultClass.
    """
    times = np.concatenate((results.t, new_result.t))
    concs = np.concatenate((results.y[0], new_result.y[0]))
    elevs = np.concatenate((results.y[1], new_result.y[1]))

    return ResultClass(times, [concs, elevs])


def simulate_elevation(water_heights, settle_rate, bulk_dens, bound_conc, org_rate=0.0, comp_rate=0.0, sub_rate=0.0, init_elev=0.0, init_conc=0.0, timestep=1.0, min_depth=0.001):
    '''
    Zero-dimensional model of elevation change on a tidal platform given an array of water heights, settling rate of the sediment,
    dry bulk density of the sediment, and boundary concentration of the tidal channel.
    '''

    # DEFINE INTERNAL FUNCTIONS

    # Function that is used by solve_ivp to turn off solver when this function becomes zero.
    def below_platform(t, init_vals, *args):
        depth = tide_spline(t) - (init_vals[1] + min_depth)
        return depth

    below_platform.terminal = True
    below_platform.direction = -1

    # Function to solve derivatives of concentration and elevation.
    def aggrade(t, init_vals, *args):

        # set initial values for concentration and elevation
        init_conc = init_vals[0]
        init_elev = init_vals[1]

        # use spline function for tide height to set current water_height
        water_height = tide_spline(t)
        depth = water_height - init_elev  #calculate current depth

        # use derivative of tide spline to get current gradient and set H
        tide_deriv = tide_spline_deriv(t)

        if tide_deriv > 0:
            H = 1
        else:
            H = 0

        delta_conc = - (settle_rate * init_conc) / depth - H / depth * (init_conc - bound_conc) * tide_deriv
        sed_rate = settle_rate * (init_conc + delta_conc) / bulk_dens
        delta_elev = sed_rate + org_rate - comp_rate - sub_rate

        return [delta_conc, delta_elev]

    # SET MODEL PARAMETERS

    # Make timeseries of water heights
    try:  # Try to infer timestep.
        water_heights.index.freq.n
        #pd.infer_freq(water_heights.index)
    except:  # If cannot, use timestep passed by function. 1s is the default.
        vals = water_heights
        print('Couldn\'t infer timestep. Using {}s timestep.'.format(timestep))
    else:  # Set timestep to infered value.
        #timestep = pd.tseries.frequencies.to_offset(pd.infer_freq(water_heights.index)).n
        timestep = water_heights.index.freq.n
        vals = water_heights.values

    index = np.arange(0, len(water_heights) * timestep, timestep)  # make numeric index in seconds using timestep
    data = pd.Series(data=vals, index=index)  # make pd.Series

    # Convert yearly linear rates to per timestep

    org_rate = org_rate / 365 / 24 / 60 / 60 * timestep
    comp_rate = comp_rate / 365 / 24 / 60 / 60 * timestep
    sub_rate = sub_rate / 365 / 24 / 60 / 60 * timestep

    # Initialize
    elev = init_elev
    conc = init_conc
    pos = 0
    results = ResultClass()

    # Progress through timeseries until all data is processed or an error is raised
    while True:
        remaining_data = data.loc[pos:]  # remaining tide data to be processed
        elev_arr = degrade_linear(init_elev=elev,
                                  time_arr=(remaining_data.index.values - remaining_data.index.values[0]),
                                  comp_rate=comp_rate, sub_rate=sub_rate)
        elev_ser = pd.Series(data=elev_arr, index=remaining_data.index)
        data_above_platform = remaining_data[remaining_data > (elev_ser + min_depth)]  # tide data that is above the platform

        # break when not enough data to make a spline
        if len(data_above_platform) < 4:
            results.t = np.append(results.t, data.index.values[-1])
            results.y[0] = np.append(results.y[0], 0)
            results.y[1] = np.append(results.y[1], elev_ser.values[-1])
            break

        pos_at_start = data_above_platform.index[0] # index value of start of next inundation
        pos_at_ends = data_above_platform.index[np.where(np.diff(data_above_platform.index) != timestep)]  # index values of end of remaining inundations

        # set end position of integration
        if len(pos_at_ends) == 0:  # Set the end position to the end of the data when the last data point ends above the platform.
            end = data_above_platform.index[-1]
        else:
            end = pos_at_ends[0]  # Set the end position to the end of the first inundation in the remaining data.

        # define subset to integrate
        subset = data_above_platform.loc[:end]
        elev = elev_ser.loc[pos_at_start]

        # ensure subset is long enough to make a spline
        if len(subset) < 4:
            pos = subset.index[-1] + 1
            elev = elev_ser.loc[pos]
            continue

        # create continuous functions for tide height and the derivative of tide height to be called by the solver
        tide_spline = InterpolatedUnivariateSpline(subset.index, subset.values)
        tide_spline_deriv = tide_spline.derivative()

        # define limits of integration
        t_span = [subset.index[0], subset.index[-1]]

        # integrate over the current inundation cycle
        aggrade_result = solve_ivp(fun=aggrade, t_span=t_span, y0=[conc, elev],
                           events=below_platform, args=(bound_conc, settle_rate, bulk_dens, min_depth, org_rate, comp_rate, sub_rate))

        # raise exception if solver fails
        if aggrade_result.status == -1:
            print('Solver failed to integrate from t={} to t={}.'.format(t_span[0], t_span[-1]))
            print(aggrade_result.message)
            break

        # concatenate data
        results = concatenate_results(results, aggrade_result)

        # update params for next run
        pos = aggrade_result.t[-1] + 1
        elev = aggrade_result.y[1][-1]

    # add start of elevation
    if results.t[0] != 0:
        results.t = np.insert(results.t, 0, 0)
        results.y[0] = np.insert(results.y[0], 0, 0)
        results.y[1] = np.insert(results.y[1], 0, init_elev)

    return results


def run_model(tides_ts, settle_rate, bulk_dens, bound_conc, org_rate=0.0, comp_rate=0.0, sub_rate=0.0, init_elev=0.0, years=1, slr=0, verbose=False):
    '''
    Wrapper for the run_one function that simulates multiple years of tidal inundation
    with linear sea level rise. Returns a pd.Series for the tides and a solve_ivp
    result-like object for the integration.
    '''
    start_time = time.perf_counter()
    # reindex input tides to use pd.timedelta_range instead of actual dates
    index = pd.timedelta_range(start=0, periods=len(tides_ts), freq=tides_ts.index.freq)
    data = pd.Series(data=tides_ts.values, index=index)

    # initialize
    timestep = tides_ts.index.freq.n
    elev = init_elev
    results = ResultClass()
    tides = pd.Series(dtype='float64')

    # loop through a specified number of years
    for year in range(0, years):
        offset = (len(data) * timestep) * year  # offset to be add to tide index and result times depending on the year

        # make pd.Series for current year of tide data
        data = pd.Series(data=data.values, index=index + pd.offsets.Second(offset))

        # run model for one year and update time vector with offset
        result = simulate_elevation(water_heights=data, settle_rate=settle_rate, bulk_dens=bulk_dens, bound_conc=bound_conc, org_rate=org_rate, comp_rate=comp_rate, sub_rate=sub_rate, init_elev=elev)
        result = ResultClass(result.t + offset, result.y)

        # concatenate results
        results = concatenate_results(results, result)
        tides = tides.append(data)
        
        # update params for next year
        elev = result.y[1][-1]
        data = data + slr

    if verbose is True:
        print('Simulation length:     {} yr'.format(years))
        print('Total accumulation:    {:.4f} m'.format((result.y[1][-1] - init_elev)))
        print('Final elevation:       {:.4f} m'.format(result.y[1][-1]))
        print('Runtime:               {}'.format(time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time))))

    return tides, results