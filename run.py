import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import tidal_marsh as tm


# base = tm.load_tide(tm.ROOT / "data" / "processed" / "tides.feather")
# base = base - 0.22

base = tm.load_tide(tm.ROOT / "data" / "processed" / "test.feather")
base = base.loc["2010"]

tides = tm.Tides(water_levels=base)

ssc = pd.DataFrame(data={"month": np.tile(np.arange(1, 13, 1), 2), "ssc": np.nan})
ssc.loc[ssc.month == 6, "ssc"] = 0.1
ssc.loc[ssc.month == 9, "ssc"] = 0.4
ssc = ssc.interpolate(method="cubicspline").loc[5 : 5 + 11].squeeze().set_index("month").ssc.sort_index()

tide = tides.calc_elevation(beta=0.01, trend=0.005)
init_elev = 2.6
slr = 0.005
beta = 0.01
conc_bound = ssc.values
grain_diam = 2.5e-5
bulk_dens = 1000
org_rate = 2e-4
comp_rate = -4e-3
sub_rate = -3e-3

model = tm.Model(
    tides=tide,
    init_elev=init_elev,
    conc_bound=conc_bound,
    grain_diam=grain_diam,
    grain_dens=2.65e3,
    bulk_dens=bulk_dens,
    org_rate=org_rate,
    comp_rate=comp_rate,
    sub_rate=sub_rate,
)

model.run()

# land_elev_init = [1.5, 2.6]
# slr = [0, 0.005, 0.01]
# beta = [0, 0.01, 0.02]
# conc_bound = [ssc * 0.5, ssc, ssc * 2]
# grain_diam = [2.5e-5 * 0.5, 2.5e-5, 2.5e-5 * 2]
# bulk_dens = [800, 1000, 1200]

# params = utils.make_combos(land_elev_init=land_elev_init, slr=slr, beta=beta, conc_bound=conc_bound, grain_diam=grain_diam, bulk_dens=bulk_dens)

# Parallel(n_jobs=15)(delayed(core.simulate)(tides=tides, params=param) for param in params)
