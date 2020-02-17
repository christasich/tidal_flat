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
    
    end_slr = int(dict['yr']) * float(dict['slr'])
    MHW = np.linspace(0, end_slr, len(data.index)) + 1.67
    data = feather.read_dataframe(file)
    data = data.set_index('Datetime')
    
    recovery_bool = data.loc[:, 'z'] >= MHW
    if recovery_bool.all() == True:
        recovery_time = None
    else:
        recovery_index = np.argwhere(np.diff(recovery_bool)).squeeze()
        recovery_date = data.index[int(np.mean(recovery_index))]
        recovery_time = (recovery_date - data.index[0]).total_seconds()
    
    dict['end_z'] = data.z[-1]
    dict['recovery_time'] = recovery_time
    
    results.append(dict)

df = pd.DataFrame.from_dict(results)
base = df.loc[df['z0'] == '2']
results_df = df.loc[df['z0'] != '2']
    
results_df = results_df.sort_values(by=['slr', 'sscfactor'], ignore_index=True)
results_df.to_csv('../data/interim/ssc_v_slr_runs.csv', index=False)