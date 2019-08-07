# %%
# ==============================================================================
# IMPORT PACKAGES
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def parser(x):
    time = pd.datetime.strptime(x, '%d-%b-%Y %H:%M:%S')
    return time


start = pd.datetime(2015, 5, 15, 1)
end = pd.datetime(2016, 5, 14, 1)
dt = 3
rep = 10
rep_end = start.replace(year=start.year + rep)
slr = 0.003

df = pd.read_csv(file, parse_dates=['datetime'],
                 date_parser=parser, index_col='datetime')
df1 = df[(df.index >= start) & (df.index < end) & (df.index.minute == 0)]
df_out = df1['pressure'] - np.mean(df1['pressure'])

tides = df_out

new_index = pd.DatetimeIndex(start=start, end=rep_end, freq='H')

tides_rep = np.tile(tides.values, rep + 1)
tides_rep = tides_rep[:len(new_index)]

slr_rep = np.linspace(0, rep * slr, num=len(tides_rep))

out = tides_rep + slr_rep

# %%
# ==============================================================================
# PARAMATERIZE AND RUN DELTA Z
# ==============================================================================

# gs = 0.03
# ws = ((gs/1000)**2*1650*9.8)/0.018
# rho = 700
# SSC = 0.2
# dP = 0
# dO = 0
# dSub = 0.002
# z0 = 0
# heads = tides
# time = tides.index
# dM = 0

# C0 = np.zeros(len(heads))
# C = np.zeros(len(heads))
# z = np.zeros(len(heads)+1)
# z[0:2] = z0
# dz = np.zeros(len(heads))
# dh = np.zeros(len(heads))
# dt = (time[1]-time[0]).total_seconds()
# j = 1
# for h in heads[1:]:
#     dh[j] = (h-heads[j-1])/dt
#     C0[j] = 0
#     if h > z[j]:
#         if dh[j] > 0:
#             C0[j] = 0.69*SSC*(h-z[j])
#             C[j] = (C0[j]*(h-heads[j-1])+C[j-1]*(h-z[j])) / \
#                 (2*h-heads[j-1]-z[j]+ws/dt)
#         else:
#             C[j] = (C[j-1]*(h-z[j]))/(h-z[j]+ws/dt)
#     else:
#         C[j] = 0
#     dz[j] = (ws*C[j]/rho)*dt
#     z[j+1] = z[j] + dz[j] + dO - dP - dM/(8760/(dt/3600))
#     j = j + 1

# z = z[1:]


# %%
start = pd.datetime(2015, 5, 15, 1)
end = pd.datetime(2016, 5, 14, 1)
dt = 3
rep = 10

gs = 0.03
ws = ((gs/1000)**2*1650*9.8)/0.018
rho = 700
SSC = 0.2
dP = 0
dO = 0
dSub = 0.002
z0 = 0
dM = 0

index = new_index
heads = out

columns = ['h', 'dh', 'C0', 'C', 'z', 'dz']
df = pd.DataFrame(index=index, columns=columns)
df[:] = 0
dt = (index[1]-index[0]).total_seconds()
df.loc[:, 'h'] = tides_rep
df.loc[:, 'dh'] = df.loc[:, 'h'].diff()/dt

def calc_c0(h, dh, z, A, ssc):
    if (h > z and dh > 0):
        return A * ssc * (h - z)
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

for t in df.index[1:]:
    t_min_1 = t - pd.Timedelta(hours=1)
    df.loc[t,'z'] = calc_z(df.at[t_min_1,'z'],df.at[t_min_1,'dz'],0,0)
    df.loc[t,'C0'] = calc_c0(df.at[t,'h'], df.at[t,'dh'], df.at[t,'z'], A, SSC)
    df.loc[t,'C'] = calc_c(df.at[t,'C0'], df.at[t,'h'],df.at[t_min_1,'h'], df.at[t,'dh'], df.at[t_min_1, 'C'], df.at[t,'z'], ws, dt)
    df.loc[t,'dz'] = calc_dz(df.at[t,'C'], ws, rho, dt)

for t in df.index[1:]:
    t1 = t
    t2 = df.index[t+1]
    if h > df.z[j]:
        if df.dh[j] > 0:
            df.loc[t1, 'C0'] = 0.69 * SSC * (h - df.z[j])
            df.C[j] = (df.C0[j] * (h - df.h[j - 1]) + df.C[j - 1] *
                       (h - df.z[j])) / (2 * h - df.h[j - 1] - df.z[j] + ws / dt)
        else:
            df.C[j] = (df.C[j - 1] * (h - df.z[j])) / (h - df.z[j] + ws / dt)
    else:
        df.C[j] = 0
    df.dz[j] = (ws * df.C[j] / rho) * dt
    df.z[j + 1] = df.z[j] + df.dz[j] + dO - dP - dM / (8760 / (dt/3600))
    j = j + 1

# %%
