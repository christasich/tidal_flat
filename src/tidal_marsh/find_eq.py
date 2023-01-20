from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import bisect, brentq, root_scalar
from joblib import Parallel, delayed

from src import tidal_marsh as tm

data = pd.read_pickle("data/processed/tides-pickle/1900-2099_5T_nodal.pickle")
tides = tm.Tides(series=data)

elevation_ref = 0.0
time_ref = tides.start

platform = tm.Platform(time_ref=tides.start, elevation_ref=0.0)

grain_diameter = 2.5e-5
bulk_density = 1e3
grain_density = 2.65e3

ssc = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
mtr = [1, 2, 3, 4, 5]
slr = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
combos = tm.helpers.make_combos(ssc=ssc, mtr=mtr, slr=slr)
df = pd.DataFrame(combos)
df["af"] = df.mtr / tides.datums.MN

# highest aggradation - 2053
# lowest aggradation - 2024
# median aggradation - 2048


@cache
def change_per_year(init_elevation, ssc, af, slr) -> float:
    tides = base.amplify(af).raise_sea_level(slr)
    model = tm.Model(
        ssc=ssc,
        grain_diameter=2.5e-5,
        grain_density=2.65e3,
        bulk_density=1e3,
        tides=tides,
        platform=tm.Platform(time_ref=tides.start, elevation_ref=init_elevation),
        pbar_opts={"disable": True},
    )
    model.run()
    return model.change_per_year - slr


def find_eq(f, a, b, xtol, args):
    try:
        return brentq(f, a, b, xtol=xtol, args=args)
    except ValueError:
        return np.nan


year = "2037"
base = tides.subset(start=year, end=year)

bounds = pd.DataFrame([base.amplify(af).datums[["LAT", "HAT"]].rename_axis("mtr") for af in df.af.unique()])
bounds.index = bounds.index + 1
df = df.join(bounds, on="mtr")


results = Parallel(n_jobs=-1, verbose=10, backend="multiprocessing")(
    delayed(find_eq)(f=change_per_year, a=row.LAT, b=row.HAT, xtol=1e-3, args=(row.ssc, row.af, row.slr))
    for row in df.itertuples()
)

df["eq"] = results
df.to_csv(f"init-{year}.csv")


# years = ["2053", "2048", "2024"]

# for year in years:
#     base = tides.subset(start=year, end=year)

#     bounds = pd.DataFrame([base.amplify(af).datums[["LAT", "HAT"]].rename_axis("mtr") for af in df.af.unique()])
#     bounds.index = bounds.index + 1
#     df2 = df.join(bounds, on="mtr")

#     results = Parallel(n_jobs=-1, verbose=10, backend="multiprocessing")(
#         delayed(find_eq)(f=change_per_year, a=row.LAT, b=row.HAT, xtol=1e-3, args=(row.ssc, row.af, row.slr))
#         for row in df2.itertuples()
#     )

#     df2["eq"] = results
#     df2.to_csv(f"init-{year}.csv")
#     change_per_year.cache_clear()
