#%% Import packages

import pandas as pd
import numpy as np
import os
import feather
import ntpath

#%%

feather_path = '../data/interim/results'
feather_files = os.listdir(feather_path)
feather_files = [os.path.abspath(os.path.join(feather_path, f)) for f in feather_files]

# %%
index = feather.read_dataframe('../data/interim/tides/tides.0.000_slr').Datetime

data = feather.read_dataframe(feather_files[10])
data = data.set_index(index)

for file in feather_files:
    name = ntpath.basename(file)
    new_name = 'yr_20'+name[5:]
    new_path = os.path.join(ntpath.dirname(file), new_name)
    os.rename(file, new_name)

# %%

