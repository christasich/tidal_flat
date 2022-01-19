import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from src import models
from src import features
from src import definitions as defs

wdir = defs.ROOT / "projects" / "tidal_flat_0d" / "data"

base = pd.read_feather(wdir / "processed" / "tides.feather").set_index("datetime").squeeze().loc[:"2059"] - 0.22
base.index = pd.DatetimeIndex(data=base.index, freq="infer")

tides = models.Tides(data=base)

ssc = pd.DataFrame(data={"month": np.tile(np.arange(1, 13, 1), 2), "ssc": np.nan})
ssc.loc[ssc.month==6, "ssc"] = 0.1
ssc.loc[ssc.month==9, "ssc"] = 0.4
ssc = ssc.interpolate(method="cubicspline").loc[5:5+11].squeeze().set_index("month").ssc.sort_index()

land_elev_init = [1.5, 2.6]
slr = [0, 0.005, 0.01]
beta = [0, 0.01, 0.02]
conc_bound = [ssc * 0.5, ssc, ssc * 2]
grain_diam = [2.5e-5 * 0.5, 2.5e-5, 2.5e-5 * 2]
bulk_dens = [800, 1000, 1200]

params = features.make_combos(land_elev_init=land_elev_init, slr=slr, beta=beta, conc_bound=conc_bound, grain_diam=grain_diam, bulk_dens=bulk_dens)

Parallel(n_jobs=15)(delayed(models.simulate)(tides=tides, params=param) for param in params)