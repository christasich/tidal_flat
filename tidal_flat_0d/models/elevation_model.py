from collections import namedtuple
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import inspect


def calc_conc(
    bound_conc, tide_height, prev_tide_height, prev_conc, elev, prev_elev, settle_rate, timestep, method='CT'
):
    depth = tide_height - elev
    prev_depth = prev_tide_height - prev_elev
    change_in_depth = depth - prev_depth

    # Checks
    tide_above_platform = depth > 0
    prev_tide_above_platform = prev_depth > 0
    depth_stable = prev_depth >= 0.0015
    tide_increasing = tide_height > prev_tide_height
    settling_valid = settle_rate * prev_conc * timestep <= prev_depth * prev_conc

    if settling_valid:
        settled_amount = settle_rate * prev_conc * timestep
    else:
        settled_amount = prev_depth * prev_conc

    if method == 'CT':

        if tide_above_platform and depth_stable:
            if tide_increasing:
                conc = prev_conc - settled_amount / prev_depth + 1 / prev_depth * (bound_conc * change_in_depth - prev_conc * change_in_depth)
                return conc
            if not tide_increasing:
                conc = prev_conc - settled_amount / prev_depth
                return conc
            else:
                raise Exception('Tide not increasing or decreasing.')
        if not tide_above_platform or not depth_stable:
            conc = 0
            return conc
        else:
            raise Exception('Error in tide_above_platform or depth_stable')

    if method == 'JG':
        if prev_tide_above_platform and tide_above_platform:
            if tide_increasing:
                conc = (prev_depth * prev_conc) / depth - settled_amount / depth + bound_conc * (1 - prev_depth / depth)
                return conc
            if not tide_increasing:
                conc = (prev_depth * prev_conc) / depth - settled_amount / depth
                return conc
            else:
                raise Exception('Tide not increasing or decreasing.')
        elif not prev_tide_above_platform and tide_above_platform:
            conc = bound_conc
            return conc
        if not tide_above_platform:
            conc = 0
            return conc
        else:
            raise Exception('Error in tide_above_platform or depth_stable')

def accumulate_sediment(conc, settle_rate, timestep):
    deposited_sediment = settle_rate * conc * timestep
    return deposited_sediment


def aggrade(start_elev, sediment, organic, compaction, subsidence):
    elev = start_elev + sediment + organic - compaction - subsidence
    return elev

def stokes_settling(grain_dia, grain_den, fluid_den = 1000, fluid_visc = 0.001, g = 9.8):
    settle_rate = (
        (2 / 9 * (grain_den - fluid_den) / fluid_visc) * g * (grain_dia / 2) ** 2
    )
    return settle_rate


def simulate_elevation(params):

    # Unpack params
    index = params.index

    if isinstance(params.bound_conc, (int, float)):
        bound_conc = np.full(len(index), params.bound_conc)
    else:
        bound_conc = params.bounc_conc

    timestep = (index[1] - index[0]).total_seconds()
    water_height = params.water_height
    settle_rate = params.settle_rate
    bulk_den = params.bulk_den

    organic_matter = params.organic_rate / 8760 / 60 / 60 * timestep
    compaction = params.compaction_rate / 8760 / 60 / 60 * timestep
    subsidence = params.subsidence_rate / 8760 / 60 / 60 * timestep

    elev = np.zeros(len(index))
    elev[0] = params.start_elev

    elev_change = np.zeros(len(index))
    conc = np.zeros(len(index))
    deposited_sediment = np.zeros(len(index))

    counter = np.arange(1, len(index))

    for t in tqdm(
        counter,
        total=len(index[1:]),
        unit="steps",
    ):
        elev[t] = aggrade(elev[t - 1], elev_change[t - 1], organic_matter, compaction, subsidence)
        conc[t] = calc_conc(
            bound_conc[t],
            water_height[t],
            water_height[t - 1],
            conc[t - 1],
            elev[t],
            elev[t - 1],
            settle_rate,
            timestep, method=params.conc_method
        )
        deposited_sediment[t] = accumulate_sediment(
            conc[t], settle_rate, timestep
        )
        elev_change[t] = deposited_sediment[t] / bulk_den

    data = pd.DataFrame(data={'elev': elev, 'water_height': water_height, 'bound_conc': bound_conc, 'conc': conc,'deposited_sediment': deposited_sediment, 'elev_change': elev_change}, index=index)
    data['depth'] = data.water_height - data.elev
    data.depth = np.where(data.depth < 0, 0, data.depth)
    data['suspended_sediment'] = data.conc * data.depth
    data['incoming_sediment'] = bound_conc * (data.depth - data.depth.shift(1))
    data.incoming_sediment = np.where(data.incoming_sediment < 0, 0, data.incoming_sediment)

    return data