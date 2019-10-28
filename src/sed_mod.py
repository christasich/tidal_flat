# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import feather
from tqdm import tqdm

# %% Functions


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


def calc_c0(h, dh, z, A, timestamp):
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


def run_model(tides, gs, rho, dP, dO, dM, A, z0):
    ws = ((gs / 1000) ** 2 * 1650 * 9.8) / 0.018
    columns = ['h', 'dh', 'C0', 'C', 'dz', 'z']
    index = tides.index
    df = pd.DataFrame(index=index, columns=columns)
    df[:] = 0
    df.iloc[:, 5][0:2] = z0
    df.loc[:, 'h'] = tides
    df.loc[:, 'dh'] = df.loc[:, 'h'].diff() / pd.Timedelta(dt).total_seconds()

    for t in tqdm(tides.index[1:], total=len(tides.index[1:]), unit='steps'):
        t_min_1 = t - pd.Timedelta(dt)
        df.loc[t, 'z'] = calc_z(df.at[t_min_1, 'z'], df.at[t_min_1, 'dz'], 0, 0)
        df.loc[t, 'C0'] = calc_c0(
            df.at[t, 'h'], df.at[t, 'dh'], df.at[t, 'z'], A, t)
        df.loc[t, 'C'] = calc_c(df.at[t, 'C0'], df.at[t, 'h'], df.at[t_min_1, 'h'],
                                df.at[t, 'dh'], df.at[t_min_1, 'C'], df.at[t, 'z'], ws, dt_sec)
        df.loc[t, 'dz'] = calc_dz(df.at[t, 'C'], ws, rho, dt_sec)
        
    return df


# %% Load Data

# Load Tides
tides_file = './data/processed/p32_tides.csv'


start = pd.datetime(2015, 5, 16, 0)
end = pd.datetime(2016, 5, 16, 0)
dt = '3H'
dt_sec = pd.Timedelta(dt).total_seconds()
rep = 50
rep_end = start.replace(year=start.year + rep)
slr = 1

tides = read_data(tides_file, start, end, dt)
tides_rep = rep_series(tides, start, end)
tides_rep_with_slr = apply_linear_slr(tides_rep, slr)

# Load weeksly ssc

ssc_file = './data/processed/ssc_by_week.csv'

ssc_by_week = pd.read_csv(ssc_file, index_col=0)

# %% Run sediment model

df = run_model(tides=tides_rep_with_slr, gs=0.03, rho=1100, dP=0, dO=0, dM=0.002, A=0.7, z0=0.65)
feather.write_dataframe(df, './data/interim/longrun')


# %%

f, ax = plt.subplots(figsize=(9,3))
sns.set_style("whitegrid")

plt_data = df.shift(freq=pd.Timedelta(days=1326))
plt_data = plt_data[::10]
sns.lineplot(data=plt_data.h, alpha = 0.25)
sns.lineplot(data=plt_data.z, color = 'black')
plt.title("Tidal Heights and Land Surface")
plt.xlabel("Year")
plt.ylabel("Height above Mean Water Level (m)")
plt.xlim(start + pd.Timedelta(days=1326), rep_end + pd.Timedelta(days=1326))
plt.savefig('fig.jpg', dpi=1000)
plt.show()