import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import tidal_marsh as tm
from joblib import Parallel, delayed
from pathlib import Path

base_path = Path('data/.cache/tides_1960-2052.pickle')

grain_diameter = 25.0e-6
grain_density = 2.65e+3
bulk_density = 1000

mtr = np.arange(1, 11)
rslr = np.arange(1, 7) / 1000

gbm_ssc = 1.0
te = np.arange(1, 11) / 10
ssc = (gbm_ssc * te)

params = tm.make_combos(base=base_path, mtr=mtr, ssc=ssc, rslr=rslr, grain_diameter=grain_diameter,
                        grain_density=grain_density, bulk_density=bulk_density)


def run(base, mtr, ssc, rslr, grain_diameter, grain_density, bulk_density, pos):
    tides = pd.read_pickle(base)

    af = mtr / tides.summary.MN
    amplified = tides.amplify_from_factor(af=af)
    slr = tides.calc_slr(rate=rslr)
    tides.update(amplified + slr)

    initial_elevation = tides.summary.MSL

    model = tm.Model(
        water_levels=tides.data.elevation,
        initial_elevation=initial_elevation,
        ssc=ssc,
        grain_diameter=grain_diameter,
        grain_density=grain_density,
        bulk_density=bulk_density,
        position=pos,
        pbar_name=f'{pos:03} | TR={mtr} m | SSC={ssc:.1f} g/L | RSLR={rslr * 1000:.1f} mm | ')
    model.initialize()
    model.run()
    model.uninitialize()
    results = process_results(tides, model)
    results['mtr'] = mtr
    results['ssc'] = ssc
    results['rslr'] = rslr
    results.to_csv(f'results/{pos:03}.csv')


def process_results(tide_cls, model):
    results = model.results
    results['elevation_change'] = results.elevation - results.elevation[0]
    results = results.resample('H').mean().interpolate().resample('A').mean()

    results['msl'] = tide_cls.data.elevation.resample('A').mean()
    results['mhw'] = tide_cls.highs.resample('A').mean()
    results['mshw'] = tide_cls.springs.highs.resample('A').mean()

    inundations = model.inundations.reset_index(drop=True).set_index('start')

    results['hydroperiod'] = inundations.hydroperiod.resample('A').mean() / pd.to_timedelta('1H')
    results['cum_hydroperiod'] = inundations.hydroperiod.resample('A').sum() / pd.to_timedelta('1H')
    results['daily_flood'] = (inundations.hydroperiod.resample('D').sum() / pd.to_timedelta('1H')).resample('A').mean()
    results['depth'] = inundations.depth.resample('A').mean()
    results['cum_depth'] = inundations.depth.resample('A').sum()

    return results


meta = pd.DataFrame.from_records(params, index='pos')
meta.to_csv('results/_metadata.csv')
Parallel(n_jobs=10)(delayed(run)(**param) for param in params)
