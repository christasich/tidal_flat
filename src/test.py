import sys
import numpy as np
import pandas as pd
from pyprojroot import here

root = here()
sys.path.append(str(root))
wdir = root / "data" / "interim"


from src import features
from src import models

tides = features.load_tide(wdir, "tides.feather")
tides = tides - (np.mean(tides) + 0.6)

dt = 1  # seconds
tides = tides.iloc[0::dt]

# Set grain parameters and settling velocity
grain_dia = 0.000035  # grain diameter (m)
grain_dens = 2650  # grain density (kg/m^3)
# Set basic model parameters
bound_conc = 1  # boundary concentration (g/L or kg/m^3)
bulk_dens = 900  # dry bulk density of sediment (kg/m^3)
min_depth = 0.001  # Minimum depth required before integrating. Used for stability at very shallow depths. (m)
init_elev = 1.15  # initial elevation (m)
init_conc = 0
years = 1  # total run length (yr)
slr = 0.005  # yearly rate of sea level rise (m)

org_rate = 0.0002
comp_rate = 0.004
sub_rate = 0.003
min_depth = 0.001

N = 1

num_tides = pd.Series(data=tides.values, index=np.arange(0, len(tides), dt))

params = models.Params(
            tides=num_tides,
            init_elev=init_elev,
            init_conc=init_conc,
            bound_conc=bound_conc,
            grain_dia=grain_dia,
            grain_dens=grain_dens,
            bulk_dens=bulk_dens,
            org_rate=org_rate,
            comp_rate=comp_rate,
            sub_rate=sub_rate,
            min_depth=min_depth)
sim = models.Simulation(
    params=params,
    N=N)

results = models.simulate_elevation(sim)