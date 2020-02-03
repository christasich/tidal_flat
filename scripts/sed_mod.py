# %%
import pandas as pd
import numpy as np
import os
import sys
import subprocess
import feather
from tqdm import tqdm
import multiprocessing as mp
import itertools
import inspect
import shutil

# %% Functions

if sys.platform == 'linux':
    os.chdir('/home/chris/projects/tidal_flat_0d/')

def read_data(file, start, end, dt):
    def parser(x):
        return pd.datetime.strptime(x, '%d-%b-%Y %H:%M:%S')
    df = pd.read_csv(file, parse_dates=[
        'datetime'], date_parser=parser, index_col='datetime')
    df1 = df[(df.index >= start) & (df.index < end)]
    resample_df = df1.resample(dt).first()
    return resample_df['pressure'] - np.mean(resample_df['pressure'])


def rep_series(df, start, end):
    freq = df.index.freq
    index = pd.DatetimeIndex(start=start, end=rep_end, freq=freq)
    values = np.tile(df.values, rep + 1)[:len(index)]
    return pd.Series(data=values, index=index)


def apply_linear_slr(df, rate_slr):
    freq = df.index.freq.delta.total_seconds()
    rate_slr_sec = rate_slr / 365 / 24 / 60 / 60
    stop = freq * len(df) * rate_slr_sec
    slr_values = np.linspace(0, stop, num=len(df))
    return df + slr_values


def make_tides(run_length, dt, slr):
    if not os.path.isfile('./data/interim/tides/tides-{0}_slr.feather'.format('%.4f' % slr)):
        Rscript = "Rscript"
        path = os.path.join(os.getcwd(),'scripts/make_tides.R')
        subprocess.run([Rscript, path, str(run_length), str(dt), '%.4f' % slr, os.getcwd()])
    
    tides = feather.read_dataframe('./data/interim/tides/tides-{0}_slr.feather'.format('%.4f' % slr))
    tides = tides.set_index('Datetime')
    
    return tides


def calc_c0(h, dh, z, A, timestamp):
    global ssc_by_week
    week = timestamp.week
    ssc = ssc_by_week.loc[week].values[0]
    if (h > z and dh > 0):
        return A * ssc
    else:
        return 0


def calc_c(c0, h, h_min_1, dh, c_min_1, z, ws, dt):
    if (h > z and dh > 0):
        return (c0 * (h-h_min_1) + c_min_1 * (h - z)) / (2 * h - h_min_1 - z + ws / dt)
    elif (h > z and dh < 0):
        return (c_min_1 * (h - z)) / (h - z + ws / dt)
    else:
        return 0


def calc_dz(c, ws, rho, dt):
    return (ws * c / rho) * dt


def calc_z(z_min_1, dz_min_1, dO, dP):
    return z_min_1 + dz_min_1 + dO - dP


def run_model(tides, gs, rho, dP, dO, dM, A, z0, n=0):
    global ssc_by_week
    dt = tides.index[1] - tides.index[0]
    dt_sec = dt.total_seconds()
    ws = ((gs / 1000) ** 2 * 1650 * 9.8) / 0.018
    columns = ['h', 'dh', 'C0', 'C', 'dz', 'z']
    index = tides.index
    df = pd.DataFrame(index=index, columns=columns)
    df[:] = 0
    df.iloc[:, 5][0:2] = z0
    df.loc[:, 'h'] = tides.pressure
    df.loc[:, 'dh'] = df.loc[:, 'h'].diff() / dt_sec
    df.loc[:, 'inundated'] = 0

    for t in tqdm(tides.index[1:], total=len(tides.index[1:]), unit='steps', position=n):
        t_min_1 = t - dt
        df.loc[t, 'z'] = calc_z(df.at[t_min_1, 'z'], df.at[t_min_1, 'dz'], 0, 0)
        df.loc[t, 'C0'] = calc_c0(df.at[t, 'h'], df.at[t, 'dh'], df.at[t, 'z'], A, t)
        df.loc[t, 'C'] = calc_c(df.at[t, 'C0'], df.at[t, 'h'], df.at[t_min_1, 'h'],
                                df.at[t, 'dh'], df.at[t_min_1, 'C'], df.at[t, 'z'], ws, dt_sec)
        df.loc[t, 'dz'] = calc_dz(df.at[t, 'C'], ws, rho, dt_sec)
        if df.loc[t, 'C0'] != 0:
            df.loc[t, 'inundated'] = 1
        
    hours_inundated = (np.sum(df['inundated']) * dt).astype('timedelta64[h]').astype(int)
    final_elevation = df.iloc[[-1]].z.values[0]
        
    return df, hours_inundated, final_elevation

def make_combos(run_length, dt, slr, ssc_factor, gs, rho, dP, dO, dM, A, z0):
    args = inspect.getfullargspec(make_combos).args
    multi_args = []
    n = 0
    for arg in args:
        if isinstance(eval(arg), (list, tuple, np.ndarray)):
            multi_args.append(arg)
    single_args = list(set(args) - set(multi_args))
    dict1 = [{'{0}'.format(j) : eval(j)} for j in single_args]
    dict2 = [dict(zip(multi_args, i)) for i in itertools.product(*[eval(x) for x in multi_args])]
    for entry2 in dict2:
        for entry1 in dict1:
            entry2.update(entry1)
        entry2.update({'n' : n})
        n = n + 1

    return dict2


def parallel_parser(in_data):
    global ssc_by_week

    n = in_data['n']
    
    # make tides
    run_length = in_data['run_length']
    dt = in_data['dt']
    slr = in_data['slr']
    
    tides = make_tides(run_length, dt, slr)
    
    # Load weeksly ssc
    ssc_factor = in_data['ssc_factor']
    
    ssc_file = './data/processed/ssc_by_week.csv'
    ssc_by_week = pd.read_csv(ssc_file, index_col=0) * ssc_factor
    
    # run model
    
    gs = in_data['gs']
    rho = in_data['rho']
    dP = in_data['dP']
    dO = in_data['dO']
    dM = in_data['dM']
    A = in_data['A']
    z0 = in_data['z0']
    
    df, hours_inundated, final_elevation = run_model(tides, gs, rho, dP, dO, dM, A, z0)
    out_name = 'yr_{0}-slr_{1}-gs_{2}-rho_{3}-sscfactor_{4}-dP_{5}-dM_{6}-A_{7}-z0_{8}.feather'.format(run_length, slr, gs, rho, ssc_factor, dP, dM, A, z0)
    feather.write_dataframe(df.reset_index(), './data/interim/results/{0}'.format(out_name))

    return n

#%% Run model

if __name__ == '__main__':

    # Clean up
    
    parallel = True

    #%% Set model paramters

    run_length = 20
    dt = '1 hour'
    slr = 0.003
    ssc_factor = 1
    gs = 0.035
    rho = 1400
    dP = 0
    dO = 0
    dM = 0.002
    A = 0.7
    z0 = 0.65
    
    
    if parallel == True:
        slr = np.round(np.arange(0.000, 0.031, 0.0025), 4)
        ssc_factor = np.round(np.arange(0.25, 3.25, 0.25), 2)
        model_runs = make_combos(run_length, dt, slr, ssc_factor, gs, rho, dP, dO, dM, A, z0)
        poolsize = 60
        chunksize = 1
        with mp.Pool(poolsize) as pool:
            num = 1
            for result in pool.imap_unordered(parallel_parser, model_runs, chunksize=chunksize):
                print('Finished model run {0} out of {1}'.format(num, len(model_runs)))
                num = num + 1
    else:
        tides = make_tides(run_length, dt, slr)
        ssc_file = './data/processed/ssc_by_week.csv'
        ssc_by_week = pd.read_csv(ssc_file, index_col=0) * ssc_factor

        df, hours_inundated, final_elevation = run_model(tides, gs, rho, dP, dO, dM, A, z0)
        out_name = 'yr_{0}-slr_{1}-gs_{2}-rho_{3}-sscfactor_{4}-dP_{5}-dM_{6}-A_{7}-z0_{8}.feather'.format(run_length, slr, gs, rho, ssc_factor, dP, dM, A, z0)
        feather.write_dataframe(df.reset_index(), './data/interim/results/{0}.feather'.format(out_name))