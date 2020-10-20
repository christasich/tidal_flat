import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import solve_ivp
import logging
logger = logging.getLogger(__name__)


def stokes_settling(grain_dia, grain_den, fluid_den = 1000.0, fluid_visc = 0.001, g = 9.8):
    settle_rate = (
        (2 / 9 * (grain_den - fluid_den) / fluid_visc) * g * (grain_dia / 2) ** 2
    )
    return settle_rate


def aggrade(water_heights, settle_rate, bulk_dens, bound_conc, init_elev=0.0, init_conc=0.0, timestep=1.0, min_depth=0.001):
    
    # DEFINE INTERNAL FUNCTIONS

    # Function that is used by solve_ivp to turn off solver when this function becomes zero.
    def below_platform(t, init_vals, *args):
        depth = tide_spline(t) - (init_vals[1] + min_depth)
        return depth

    below_platform.terminal = True
    below_platform.direction = -1

    # Function to solve derivatives of concentration and elevation.
    def solve_odes(t, init_vals, *args):

        # set initial values for concentration and elevation
        init_conc = init_vals[0]
        init_elev = init_vals[1]

        # use spline function for tide height to set current water_height
        water_height = tide_spline(t)
        depth = water_height - init_elev #calculate current depth

        # use derivative of tide spline to get current gradient and set H
        tide_deriv = tide_spline_deriv(t)

        if tide_deriv > 0:
            H = 1
        else:
            H = 0

        delta_conc = - (settle_rate * init_conc) / depth - H / depth * (init_conc - bound_conc) * tide_deriv
        delta_elev = settle_rate * (init_conc + delta_conc) / bulk_dens

        return [delta_conc, delta_elev]
    
    # SET MODEL PARAMETERS
    
    # Make timeseries of water heights
    try:  # Try to infer timestep. 
        pd.infer_freq(water_heights.index)
    except: # If cannot, use timestep passed by function. 1s is the default.
        vals = water_heights
        print('Couldn\'t infer timestep. Using {}s timestep.'.format(timestep))
    else: # Set timestep to infered value.
        timestep = pd.Timedelta(pd.infer_freq(water_heights.index)).total_seconds()
        vals = water_heights.values
    
    index = np.arange(0, len(water_heights) * timestep, timestep) # make numeric index in seconds using timestep
    data = pd.Series(data=vals, index=index) # make pd.Series

    # Initialize
    elev = init_elev
    conc = init_conc
    pos = 0
    elevs = np.empty(0)
    concs = np.empty(0)
    times = np.empty(0)
    inundations = 0

    # Progress through timeseries until all data is processed or an error is raised.
    while True:
        remaining_data = data.loc[pos:] # remaining tide data
        data_above_platform = remaining_data[remaining_data > (elev + min_depth)] # tide data that is above the platform
        pos_at_ends = data_above_platform.index[np.where(np.diff(data_above_platform.index) != timestep)] # index values of end of inundations relative to current elevation

        # break when not enough data to make a spline
        if len(data_above_platform) < 4:
            break

        # set end position of integration
        if len(pos_at_ends) == 0:
            end = data_above_platform.index[-1]
        else:
            end = pos_at_ends[0]

        # define subset to integrate
        subset = data_above_platform.loc[:end]

        # ensure subset is long enough to make a spline
        if len(subset) < 4:
            continue

        # create continuous functions for tide height and the derivative of tide height to be called by the solver
        tide_spline = InterpolatedUnivariateSpline(subset.index, subset.values)
        tide_spline_deriv = tide_spline.derivative()

        # define limits of integration
        t_span = [subset.index[0], subset.index[-1]]
        
        # integrate over the current inundation cycle
        result = solve_ivp(fun=solve_odes, t_span=t_span, y0=[conc, elev], events=below_platform,
                           args=(bound_conc, settle_rate, bulk_dens, min_depth))
        
        # raise exception if solver fails
        if result.status == -1:
            raise Exception(result.message)

        # concatenate data
        times = np.concatenate((times, result.t))
        concs = np.concatenate((concs, result.y[0]))
        elevs = np.concatenate((elevs, result.y[1]))

        # update params for next run
        elev = result.y[1][-1]
        pos = subset.index[-1] + 1
        inundations += 1
        
    df = pd.DataFrame(data={'concentration': concs, 'elevation': elevs}, index=times)

    return df

def run_model(tides_ts, settle_rate, bulk_dens, bound_conc, init_elev=0.0, years=1, slr=0):
    
    tides_reindex = pd.timedelta_range(start=0, periods=len(tides_ts), freq=tides_ts.index.freq)
    tides_ts = pd.Series(data=tides_ts.values, index=tides_reindex)
    data = tides_ts
    
    df = pd.DataFrame()
    timestep = pd.Timedelta(tides_ts.index.freq).total_seconds()
    elev = init_elev
    for year in range(0, years):
        offset = (len(data) * timestep) * year
        
        data = data + slr
        data_index = pd.timedelta_range(start=pd.offsets.Second(offset), periods=len(data), freq=data.index.freq)
        data = pd.Series(data=data.values, index=data_index)
        
        result = aggrade(water_heights=data, settle_rate=settle_rate, bulk_dens=bulk_dens, bound_conc=bound_conc, init_elev=elev)
        result = result.set_index(result.index + offset)
        
        elev = result.elevation.iloc[-1]
        
        df = df.append(result)
        tides_ts = tides_ts.append(data)
        
    print('Total accumulation: {:f} m'.format((df.elevation.iloc[-1] - init_elev)))
    print('Final Elevation: {:.4f} m'.format(df.elevation.iloc[-1]))
        
    return df, tides_ts