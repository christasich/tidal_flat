import numpy as np
import pandas as pd
import tidal_marsh as tm
from joblib import Parallel, delayed
from pathlib import Path

grain_diameter = 25.0e-6
grain_density = 2.65e3
bulk_density = 1000

mtr = np.arange(1, 11)
rslr = np.arange(0, 11) / 1000

gbm_ssc = 1.0
te = np.arange(1, 11) / 10
ssc = gbm_ssc * te

params = tm.make_combos(
    mtr=mtr, ssc=ssc, rslr=rslr, grain_diameter=grain_diameter, grain_density=grain_density, bulk_density=bulk_density
)


def run(mtr, ssc, rslr, grain_diameter, grain_density, bulk_density, pos):
    path = Path(f"data/.cache/tides/mtr-{mtr:.1f}_rslr-{rslr:.3f}.pickle")
    tides = pd.read_pickle(path)

    initial_elevation = tides.summary.MSL

    model = tm.Model(
        water_levels=tides.data.elevation,
        initial_elevation=initial_elevation,
        ssc=ssc,
        grain_diameter=grain_diameter,
        grain_density=grain_density,
        bulk_density=bulk_density,
        position=pos,
        pbar_name=f"{pos:04} | TR={mtr} m | SSC={ssc:.1f} g/L | RSLR={rslr * 1000:.1f} mm | ",
    )
    model.initialize()
    model.run()
    model.uninitialize()
    results = process_results(tides, model)
    results["mtr"] = mtr
    results["ssc"] = ssc
    results["rslr"] = rslr
    results.to_csv(f"results/{pos:04}.csv")


def process_results(tide_cls, model):
    results = model.results
    results["elevation_change"] = results.elevation - results.elevation[0]
    results = results.resample("H").mean().interpolate().resample("A").mean()

    results["msl"] = tide_cls.data.elevation.resample("A").mean()
    results["mhw"] = tide_cls.highs.resample("A").mean()
    results["mshw"] = tide_cls.springs.highs.resample("A").mean()

    inundations = model.inundations.reset_index(drop=True).set_index("start")

    results["hydroperiod"] = inundations.hydroperiod.resample("A").mean() / pd.to_timedelta("1H")
    results["cum_hydroperiod"] = inundations.hydroperiod.resample("A").sum() / pd.to_timedelta("1H")
    results["daily_flood"] = (inundations.hydroperiod.resample("D").sum() / pd.to_timedelta("1H")).resample("A").mean()
    results["depth"] = inundations.depth.resample("A").mean()
    results["cum_depth"] = inundations.depth.resample("A").sum()

    return results


meta = pd.DataFrame.from_records(params, index="pos")
meta.to_csv("results/_metadata.csv")

p = Path("results").glob("*")
runs = [int(x.stem) for x in p if x.is_file() and x.stem.isdigit()]
meta = pd.read_csv("results/_metadata.csv")
missing = [r for r in meta.pos if r not in runs]
missing = meta.loc[missing]
missing = missing.to_dict(orient="records")

Parallel(n_jobs=25, backend="multiprocessing")(delayed(run)(**params) for params in missing)
