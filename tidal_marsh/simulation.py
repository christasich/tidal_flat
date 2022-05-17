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

from . import tides
from . import utils
from .core import Model
from .tides import Tides, load_tides

hostname = socket.gethostname()

position = mp.Value('i', 0)
pos = None

def cache_tide(cache_dir, z2100, b, beta, k):
    filename = f"tides.z2100_{z2100}.b_{b}.beta_{beta}.k_{k}.pickle"
    path = cache_dir / filename
    if path.exists():
        logger.trace(f"Skipping. {filename} already in cache.")
        return
    with open(cache_dir / "base.pickle", "rb") as file:
        base = pickle.load(file)
    if beta == 0.0 and np.isnan(k):
        water_levels = base.data.elevation + base.calc_slr(z2100=z2100, b=b)
    else:
        water_levels = base.data.elevation + base.calc_slr(z2100=z2100, b=b) + base.calc_amplification(beta=beta, k=k)
    water_levels.to_frame(name="elvevation").squeeze().to_pickle(path)
    logger.debug(f"Done. {filename} cached successfully.")

# def reshape_ssc(self):
#     ssc = pd.DataFrame.from_records(self.config.platform.ssc, index="month").reindex(np.arange(1, 13, 1))
#     ssc = pd.concat([ssc] * 2).reset_index().interpolate(method="cubicspline")
#     ssc = ssc.loc[5 : 5 + 11].set_index("month").squeeze().sort_index()
#     self.config.platform.ssc = ssc
#     logger.info("Reshaped and interpolated SSC to monthly frequency.")

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
        logger.trace(f'Initializing worker # {pos}.')
    with logger.contextualize(id=f"{id:04}"):
        model = Model(
            water_levels=pd.read_pickle(tide),
            initial_elevation=initial_elevation,
            ssc=ssc,
            grain_diameter=grain_diameter,
            grain_density=grain_density,
            bulk_density=bulk_density,
            organic_rate=organic_rate,
            compaction_rate=compaction_rate,
            subsidence_rate=subsidence_rate,
            id=id,
            position=pos,
        )
        model.run()
        save_path = wdir / "data"
        save_path.mkdir(exist_ok=True)
        model.results.to_csv(save_path / f"{id:04}.csv")
        if len(model.invalid_inundations) > 0:
            invalid_path = wdir / 'data' / 'invalid'
            invalid_path.mkdir(exist_ok=True)
            for i in model.invalid_inundations:
                del i.logger
                name = f'{id:04}_{i.start:%Y-%m-%d}.pickle'
                with open(invalid_path / name, 'wb') as f:
                    pickle.dump(i, f, pickle.HIGHEST_PROTOCOL)
    logger.trace(f"Worker #{pos} completed run #{id:04}.")


@dataclass
class Simulations:
    config_path: InitVar[str]
    config: Bunch = field(init=False)
    wdir: Path = field(init=False)
    cache_path: Path = field(init=False)
    base_path: Path = field(init=False)
    base_size: float = field(init=False)
    cached_tides: list = field(init=False, default_factory=list)
    n_cores: int = 1
    tide_params: Bunch = field(init=False)
    platform_params: Bunch = field(init=False)
    metadata: pd.DataFrame = field(init=False)
    pbar: tqdm = field(init=False, repr=False)
    results: list = field(init=False, default_factory=list)

    @logger.contextualize(id='MAIN', server=hostname)
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
        self.wdir.mkdir(exist_ok=True, parents=True)
        if self.config.cache.rebuild and self.cache_path.exists():
            shutil.rmtree(self.cache_path)
        self.cache_path.mkdir(exist_ok=True)

    def make_combos(self):
        self.tide_params = utils.make_combos(**self.config.tides.slr, **self.config.tides.amp)
        if 0 in self.config.tides.amp['beta']:
            self.tide_params = [d for d in self.tide_params if d['beta'] != 0]
            self.tide_params = self.tide_params + utils.make_combos(**self.config.tides.slr, **{'beta': [0.0], 'k': [np.nan]})
        self.platform_params = utils.make_combos(**self.config.platform)
        logger.info(
            f"Made combination of parameters for a {len(self.tide_params)} different tides and"
            f" {len(self.platform_params)} different platforms for a total of {len(self.tide_params) * len(self.platform_params)} runs."
        )

    def configure_parallel(self, mem_per_core, n_jobs=np.nan):
        logger.info(f"Allocating {mem_per_core / 1024 ** 3:.2f} GB of RAM per core.")
        used_mem = psutil.Process(os.getpid()).memory_info().rss
        available_mem = psutil.virtual_memory().available
        logger.info(f"RAM - Available: {available_mem / 1024 ** 3:.2f} GB, Used: {used_mem / 1024 ** 3:.2f} GB")
        max_cores = available_mem / mem_per_core
        self.n_cores = int(min(self.config.parallel.max_cores, max_cores, n_jobs))
        logger.info(
            f"Max workers set to {self.n_cores}. Expected memory usage per core: ~{self.n_cores * mem_per_core / 1024 ** 3:.2f} GB"
        )

    def build_base(self):
        logger.info(f"Loading base tides from pickle file at {self.base_path}.")
        base = pd.read_pickle(self.base_path)
        self.base_size = base.memory_usage()
        logger.info(f"Base tide is {self.base_size / 1024 ** 3:.2f} GB.")
        base_pickle = self.cache_path / "base.pickle"
        if not base_pickle.exists():
            logger.info(f"Tide object not in cache. Creating from base tides.")
            tide_obj = tides.Tides(water_levels=base)
            logger.info(f"Caching base tides.")
            with open(base_pickle, "wb") as file:
                pickle.dump(tide_obj, file, pickle.HIGHEST_PROTOCOL)
        else:
            logger.info("Tide object already in cache.")

    def build_cache(self):
        def callback(result):
            self.pbar.update()

        freeze_support()
        tqdm.set_lock(RLock())

        self.configure_parallel(mem_per_core=self.base_size * 7, n_jobs=len(self.tide_params))
        logger.info(f"Building cache for {len(self.tide_params)} different tides.")
        results = []
        self.pbar = tqdm(desc="PREPARING TIDES", total=len(self.tide_params), leave=True, unit="tide", dynamic_ncols=True)
        pool = mp.Pool(processes=self.n_cores, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
        for params in self.tide_params:
            kwds = {"cache_dir": self.cache_path, **params}
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
        self.metadata['tide'] = self.metadata.apply(lambda row: (self.cache_path / f"tides.z2100_{row.z2100}.b_{row.b}.beta_{row.beta}.k_{row.k:.0f}.pickle").as_posix(), axis=1)
        self.metadata['result'] = [(self.wdir / 'data' / f'{i:04}.csv').as_posix() for i in self.metadata.index]
        self.metadata.to_csv(self.wdir / "metadata.csv")

    @logger.contextualize(id='MAIN', server=hostname)
    def setup(self):
        self.wdir = self.config.workspace.path / time.strftime("%Y-%m-%d_%H.%M.%S")
        self.cache_path = self.config.cache.path
        self.base_path = self.config.tides.path
        self.setup_workspace()
        self.configure_logging()
        self.make_combos()
        self.write_metadata()

    @logger.contextualize(id='MAIN', server=hostname)
    def prepare_cache(self):
        self.build_base()
        self.build_cache()

    @logger.contextualize(id='MAIN', server=hostname)
    def run(self, param_df=None):
        if param_df is None:
            param_df = self.metadata

        def init(lock):
            tqdm.set_lock(lock)
            tqdm.write('')

        def callback(result):
            self.pbar.update()

        freeze_support()
        tqdm.set_lock(RLock())

        self.configure_parallel(mem_per_core=self.base_size * 1.1, n_jobs=len(param_df))
        results = []
        id = 0
        self.pbar = tqdm(desc="RUNNING MODELS", total=len(param_df), leave=True, smoothing=0, unit="model", dynamic_ncols=True)
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