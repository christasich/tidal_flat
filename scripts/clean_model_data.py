#%% Import packages

import pandas as pd
import numpy as np
import os
import feather
import ntpath
from datetime import timedelta

#%%

feather_path = '../data/interim/results'
feather_files = os.listdir(feather_path)
feather_files = [os.path.abspath(os.path.join(feather_path, f)) for f in feather_files if f.endswith('.feather')]

# %%

results = []
for file in feather_files:
    
    name = ntpath.basename(file)[:-8]
    vars = name.split('-')
    
    dict = {}
    for var in vars:
        var_name = var.split('_')[0]
        var_val = var.split('_')[1]
        dict[var_name] = var_val
        
    data = feather.read_dataframe(file)
    data = data.set_index('Datetime')
    dict['end_z'] = data.z[-1]
    start = data.index[-1] - timedelta(days=365)
    end = data.index[-1]
    dt = (data.index[1]-data.index[0]).total_seconds() / 60 / 60
    dict['in_cum_hours'] = np.sum(data.inundated[start:end])
    dict['in_cum_depth'] = np.sum(data.inundation_depth[start:end])
    dict['in_max_depth'] = np.max(data.inundation_depth[start:end])
    
    results.append(dict)

df = pd.DataFrame.from_dict(results)
base = df.loc[df['z0'] == '2']
results_df = df.loc[df['z0'] != '2']
    
results_df.loc[:,'in_anomaly_hour'] = results_df['in_cum_hours'] - base['in_cum_hours'].values
results_df.loc[:,'in_anomaly_depth'] = results_df['in_cum_depth'] - base['in_cum_depth'].values
results_df.loc[:,'in_anomaly_max_depth'] = results_df['in_max_depth'] - base['in_max_depth'].values
results_df = results_df.sort_values(by=['slr', 'sscfactor'], ignore_index=True)
results_df.to_csv('../data/interim/ssc_v_slr_runs.csv', index=False)