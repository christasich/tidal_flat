import multiprocessing as mp
from multiprocessing import Pool, RLock, freeze_support
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path
import csv
from sklearn.utils import Bunch
from dataclasses import InitVar, dataclass, field
import socket

import numpy as np
import pandas as pd
import psutil
from loguru import _Logger, logger
from tqdm.auto import tqdm
import itertools as it
import yaml
import json

from .core import tides
from . import helpers
from .core import Model
from .core.tides import Tides, find_pv

hostname = socket.gethostname()

position = mp.Value("i", 0)
pos = None


def to_string(**kwargs):
    return ".".join([f"{k}_{v}" for k, v in kwargs.items()])


def cache_tide(cache_dir, base_path, mtr, rslr):
    filename = f"mtr-{mtr:.1f}_rslr-{rslr:.3f}.pickle"
    path = cache_dir / filename
    if path.exists():
        logger.trace(f"Skipping. {filename} already in cache.")
        return
    with open(base_path, "rb") as file:
        base = pickle.load(file)
    af = mtr / base.summary.MN
    water_levels = base.amplify_from_factor(af=af) + base.calc_slr(rate=rslr)
    water_levels.to_frame(name="elvevation").squeeze().to_pickle(path)
    logger.debug(f"Done. {filename} cached successfully.")


# def cache_tide(cache_dir, base_path, slr_kwargs, amp_kwargs):
#     filename = 'tides.' + to_string(**slr_kwargs) + '.' + to_string(**amp_kwargs) + '.pickle'
#     path = cache_dir / filename
#     if path.exists():
#         logger.trace(f"Skipping. {filename} already in cache.")
#         return
#     with open(base_path, 'rb') as file:
#         base = pickle.load(file)
#     if amp_kwargs['beta'] == 0.0 and np.isnan(amp_kwargs['k']):
#         water_levels = base.data.elevation + base.calc_slr(**slr_kwargs)
#     else:
#         water_levels = base.data.elevation + base.calc_slr(**slr_kwargs) + base.calc_amplification(**amp_kwargs)
#     water_levels.to_frame(name="elvevation").squeeze().to_pickle(path)
#     logger.debug(f"Done. {filename} cached successfully.")

# def reshape_ssc(self):
#     ssc = pd.DataFrame.from_records(self.config.platform.ssc, index="month").reindex(np.arange(1, 13, 1))
#     ssc = pd.concat([ssc] * 2).reset_index().interpolate(method="cubicspline")
#     ssc = ssc.loc[5 : 5 + 11].set_index("month").squeeze().sort_index()
#     self.config.platform.ssc = ssc
#     logger.info("Reshaped and interpolated SSC to monthly frequency.")
def summarize_tide(data):
    highs, lows = find_pv(data=data, window="8H")
    high_roll = highs.rolling(window=2).mean().dropna().resample("5T").mean().interpolate().dropna()
    low_roll = lows.rolling(window=2).mean().dropna().resample("5T").mean().interpolate().dropna()
    tr = high_roll - low_roll

    # # Find springs and neaps
    springs, neaps = find_pv(data=tr, window="11D")

    msl = data.resample("A").mean()
    mnhw = high_roll.loc[neaps.index].resample("A").mean()
    mhw = highs.resample("A").mean()
    mshw = high_roll.loc[springs.index].resample("A").mean()
    hat = data.resample("A").max().rolling(window=pd.to_timedelta("6798.383D") * 2, center=True).mean()

    return pd.concat([msl, mnhw, mhw, mshw, hat], keys=["msl", "mnhw", "mhw", "mshw", "hat"], axis=1)


def process_results(model):
    p_result = model.results.drop(columns="subsidence")
    p_result = p_result.resample("H").mean().interpolate().resample("A").mean()

    t_result = summarize_tide(model.water_levels)

    results = pd.concat([p_result, t_result], axis=1)

    results["rmsl"] = results.elevation - results.msl
    results["rmnhw"] = results.elevation - results.mnhw
    results["rmhw"] = results.elevation - results.mhw
    results["rmshw"] = results.elevation - results.mshw
    results["rhat"] = results.elevation - results.hat

    inundations = model.inundations.reset_index(drop=True).set_index("start")

    results["mean_hp"] = inundations.hydroperiod.resample("A").mean() / pd.to_timedelta("1H")
    results["cum_hp"] = inundations.hydroperiod.resample("A").sum() / pd.to_timedelta("1H")
    results["mean_depth"] = inundations.depth.resample("A").mean()
    results["cum_depth"] = inundations.depth.resample("A").sum()

    return results


def simulate(
    tide,
    wdir,
    id,
    initial_elevation,
    ssc,
    grain_diameter,
    grain_density,
    bulk_density,
    organic_rate,
    compaction_rate,
    subsidence_rate,
):
    global pos
    if not pos:
        with position.get_lock():
            position.value += 1
            pos = position.value
        logger.trace(f"Initializing worker # {pos}.")
    with logger.contextualize(id=f"{id:04}"):
        mtr = float(tide.split("/")[-1].split("_")[0].split("-")[1])
        rslr = float(tide.split("/")[-1].split("_")[1].split("-")[1][:5])
        model = Model(
            tides=pd.read_pickle(tide),
            init_elevation=initial_elevation,
            ssc=ssc,
            grain_diameter=grain_diameter,
            grain_density=grain_density,
            bulk_density=bulk_density,
            org_sed=organic_rate,
            compaction=compaction_rate,
            deep_sub=subsidence_rate,
            id=id,
            position=pos,
            pbar_name=f"W{pos:02}R{id:04} | TR={mtr:.1f} m | SSC={ssc:.3f} g/L | RSLR={rslr * 1000:.1f} mm | ",
        )
        model.initialize()
        model.run()
        model.uninitialize()
        model.platform.to_csv(wdir / "raw" / f"{id:04}.csv")
        processed = process_results(model)
        processed["mtr"] = mtr
        processed["ssc"] = ssc
        processed["rslr"] = rslr
        processed.to_csv(wdir / "processed" / f"{id:04}.csv")
        if len(model._cycles_invalid) > 0:
            for i in model._cycles_invalid:
                del i.logger
                name = f"{id:04}_{i.start:%Y-%m-%d}.pickle"
                with open(wdir / "invalid" / name, "wb") as f:
                    pickle.dump(i, f, pickle.HIGHEST_PROTOCOL)
    logger.trace(f"Worker #{pos} completed run #{id:04}.")


@dataclass
class Simulations:
    config_path: InitVar[str]
    config: Bunch = field(init=False)
    wdir: Path = field(init=False)
    cache_path: Path = field(init=False)
    wl_data: Path = field(init=False)
    wl_size: float = field(init=False)
    base_path: Path = field(init=False)
    cached_tides: list = field(init=False, default_factory=list)
    n_cores: int = 1
    tide_params: Bunch = field(init=False)
    platform_params: Bunch = field(init=False)
    metadata: pd.DataFrame = field(init=False)
    pbar: tqdm = field(init=False, repr=False)
    results: list = field(init=False, default_factory=list)

    @logger.contextualize(id="MAIN", server=hostname)
    def __post_init__(self, config_path):
        self.load_config(config_path)

    def load_config(self, path):
        def hook(d):
            return Bunch(**{k: Path(v) if "path" in k else v for k, v in d.items()})

        with open(path) as file:
            path = yaml.safe_load(file)
            dump = json.dumps(path)

        config = json.loads(dump, object_hook=hook)

        config.parallel.max_cores = min(config.parallel.max_cores, mp.cpu_count())

        # ssc = pd.DataFrame.from_records(config.platform.ssc, index="month").reindex(np.arange(1, 13, 1))
        # ssc = pd.concat([ssc] * 2).reset_index().interpolate(method="cubicspline")
        # ssc = ssc.loc[5 : 5 + 11].set_index("month").squeeze().sort_index()
        # config.platform.ssc = ssc
        self.config = config

    def configure_logging(self):
        if self.config.logging.enabled:

            def formatter(record):
                start = (
                    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <red>{elapsed}</red> | <magenta>{process}</magenta> |"
                    " <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                )
                end = "<level>{message}</level>\n{exception}"
                if "model_time" in record["extra"]:
                    end = "<green>{extra[model_time]}</green> | " + end
                if "id" in record["extra"]:
                    end = "<light-blue>{extra[id]}</light-blue> | " + end
                return start + end

            logger.enable("tidal_marsh")
            logger.remove()

            for handler in self.config.logging.handlers:
                if handler.enabled:
                    handler.config.enqueue = self.config.parallel.enabled
                    if handler.type == "stderr":
                        handler.config.sink = lambda msg: tqdm.write(msg, end="")
                        logger.add(**handler.config, format=formatter)
                    if handler.type == "file":
                        handler.config.sink = self.wdir / "log.json"
                        logger.add(**handler.config)

    def setup_workspace(self):
        if self.config.workspace.overwrite:
            shutil.rmtree(self.config.workspace.path)
        [(self.wdir / p).mkdir(exist_ok=True, parents=True) for p in ["raw", "processed", "invalid"]]
        if self.config.cache.rebuild and self.cache_path.exists():
            shutil.rmtree(self.cache_path)
        self.cache_path.mkdir(exist_ok=True)

    def make_combos(self):
        self.tide_params = helpers.make_combos(**self.config.tides.params)
        self.platform_params = helpers.make_combos(**self.config.platform)

        logger.info(
            f"Made combination of parameters for a {len(self.tide_params)} different tides and"
            f" {len(self.platform_params)} different platforms for a total of"
            f" {len(self.tide_params) * len(self.platform_params)} runs."
        )

    def configure_parallel(self, mem_per_core, n_jobs=np.nan):
        logger.info(f"Allocating {mem_per_core / 1024 ** 3:.2f} GB of RAM per core.")
        used_mem = psutil.Process(os.getpid()).memory_info().rss
        available_mem = psutil.virtual_memory().available
        logger.info(f"RAM - Available: {available_mem / 1024 ** 3:.2f} GB, Used: {used_mem / 1024 ** 3:.2f} GB")
        max_cores = available_mem / mem_per_core
        self.n_cores = int(min(self.config.parallel.max_cores, max_cores, n_jobs))
        logger.info(
            f"Max workers set to {self.n_cores}. Expected memory usage per core:"
            f" ~{self.n_cores * mem_per_core / 1024 ** 3:.2f} GB"
        )

    def build_base(self):
        logger.info(f"Loading base tides from pickle file at {self.wl_data}.")
        base = pd.read_pickle(self.wl_data)
        self.wl_size = base.memory_usage()
        logger.info(f"Base tide is {self.wl_size / 1024 ** 3:.2f} GB.")
        self.base_path = self.cache_path / self.wl_data.name.replace("tides", "base")
        if not self.base_path.exists():
            logger.info(f"Tide object not in cache. Creating from base tides.")
            tide_obj = tides.Tides(water_levels=base)
            logger.info(f"Caching base tides.")
            with open(self.base_path, "wb") as file:
                pickle.dump(tide_obj, file, pickle.HIGHEST_PROTOCOL)
        else:
            logger.info("Tide object already in cache.")

    def build_cache(self):
        def callback(result):
            self.pbar.update()

        freeze_support()
        tqdm.set_lock(RLock())

        self.configure_parallel(mem_per_core=self.wl_size * 7, n_jobs=len(self.tide_params))
        logger.info(f"Building cache for {len(self.tide_params)} different tides.")
        results = []
        self.pbar = tqdm(
            desc="PREPARING TIDES", total=len(self.tide_params), leave=True, unit="tide", dynamic_ncols=True
        )
        pool = mp.Pool(processes=self.n_cores, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
        for params in self.tide_params:
            kwds = {
                "cache_dir": self.cache_path,
                "base_path": self.base_path.as_posix(),
                "mtr": params.mtr,
                "rslr": params.rslr,
            }
            results.append(pool.apply_async(func=cache_tide, kwds=kwds, callback=callback))
        pool.close()
        pool.join()
        self.pbar.close()
        self.cached_tides = [r.get() for r in results]
        logger.info(f"{sum([r.successful() for r in results])}/{len(results)} tides cached successfully.")

    def write_metadata(self):
        self.metadata = [{**d[0], **d[1]} for d in it.product(self.tide_params, self.platform_params)]
        self.metadata = pd.DataFrame.from_records(self.metadata)
        self.metadata.index.name = "id"
        self.metadata["tide"] = self.metadata.apply(
            lambda row: (self.cache_path / f"mtr-{row.mtr:.1f}_rslr-{row.rslr:.3f}.pickle").as_posix(),
            axis=1,
        )
        self.metadata["raw"] = [(self.wdir / "raw" / f"{i:04}.csv").as_posix() for i in self.metadata.index]
        self.metadata["processed"] = [(self.wdir / "processed" / f"{i:04}.csv").as_posix() for i in self.metadata.index]
        self.metadata.to_csv(self.wdir / "metadata.csv")

    @logger.contextualize(id="MAIN", server=hostname)
    def setup(self):
        self.wdir = self.config.workspace.path / time.strftime("%Y-%m-%d_%H.%M.%S")
        self.cache_path = self.config.cache.path
        self.wl_data = self.config.tides.path
        self.setup_workspace()
        self.configure_logging()
        self.make_combos()
        self.write_metadata()

    @logger.contextualize(id="MAIN", server=hostname)
    def prepare_cache(self):
        self.build_base()
        self.build_cache()

    @logger.contextualize(id="MAIN", server=hostname)
    def run(self, param_df=None):
        if param_df is None:
            param_df = self.metadata

        def init(lock):
            tqdm.set_lock(lock)
            # tqdm.write("")

        def callback(result):
            self.pbar.update()

        freeze_support()
        tqdm.set_lock(RLock())

        self.configure_parallel(mem_per_core=self.wl_size * 1.1, n_jobs=len(param_df))
        results = []
        id = 0
        self.pbar = tqdm(
            desc="RUNNING MODELS", total=len(param_df), leave=True, smoothing=0, unit="model", dynamic_ncols=True
        )
        pool = mp.Pool(processes=self.n_cores, initializer=init, initargs=(tqdm.get_lock(),))
        for row in param_df.itertuples():
            kwds = {
                "tide": row.tide,
                "wdir": self.wdir,
                "id": row.Index,
                "initial_elevation": row.initial_elevation,
                "ssc": row.ssc,
                "grain_diameter": row.grain_diameter,
                "grain_density": row.grain_density,
                "bulk_density": row.bulk_density,
                "organic_rate": row.organic_rate,
                "compaction_rate": row.compaction_rate,
                "subsidence_rate": row.subsidence_rate,
            }
            results.append(pool.apply_async(func=simulate, kwds=kwds, callback=callback))
            id += 1
        pool.close()
        pool.join()
        self.pbar.close()
