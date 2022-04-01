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

import numpy as np
import pandas as pd
import psutil
from loguru import _Logger, logger
from tqdm.auto import tqdm
import os
import itertools as it
import yaml
import json

from . import tides
from . import utils
from .core import Model
from .tides import Tides, load_tides


def init():
    tqdm.write(s=" ", end="")


def cache_tide(cache_dir, z2100, b, beta, k):
    filename = f"tides.z2100_{z2100}.b_{b}.beta_{beta}.k_{k}.pickle"
    path = cache_dir / filename
    if path.exists():
        logger.info(f"Skipping. {filename} already in cache.")
        return path
    with open(cache_dir / "base.pickle", "rb") as file:
        base = pickle.load(file)
    water_levels = base.data.elevation + base.calc_slr(z2100=z2100, b=b) + base.calc_amplification(beta=beta, k=k)
    water_levels.to_frame(name="elvevation").squeeze().to_pickle(path)
    logger.info(f"Done. {filename} cached successfully.")
    return path


def simulate(
    tide,
    initial_elevation,
    ssc,
    grain_diameter,
    grain_density,
    bulk_density,
    organic_rate,
    compaction_rate,
    subsidence_rate,
    wdir,
    id=0,
):
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
        )
        model.run()
        result_path = wdir / "results" / f"{id:04}.csv"
        model.results.to_csv(result_path)

        if model.overextraction > 0.01:
            logger.warning(f"Model overextraction > 1 cm. Total: {model.overextraction * 100:.2f} cm.")
        with open(wdir / "_overextraction.csv", "a") as file:
            writer = csv.writer(file)
            writer.writerow([f"{id:04}", model.overextraction])
    logger.info(f"Done. Results saved to {result_path}")


@dataclass
class Simulations:
    config_path: InitVar[str]
    config: Bunch = field(init=False)
    wdir: Path = field(init=False)
    cache_path: Path = field(init=False)
    base_path: Path = field(init=False)
    base_size: float = field(init=False)
    cached_tides: list = field(init=False, default_factory=list)
    logger: _Logger = field(init=False)
    n_cores: int = 1
    tide_params: Bunch = field(init=False)
    platform_params: Bunch = field(init=False)
    n: int = field(init=False)
    metadata: pd.DataFrame = field(init=False)
    pbar: tqdm = field(init=False, repr=False)
    results: list = field(init=False, default_factory=list)

    def __post_init__(self, config_path):
        self.load_config(config_path)
        self.wdir = self.config.workspace.path
        self.cache_path = self.config.cache.path
        self.base_path = self.config.tides.path

        self.setup_workspace()
        self.configure_logging()
        self.reshape_ssc()
        self.make_combos()

    def load_config(self, path):
        def hook(d):
            return Bunch(**{k: Path(v) if "path" in k else v for k, v in d.items()})

        with open(path) as file:
            path = yaml.safe_load(file)
            dump = json.dumps(path)

        self.config = json.loads(dump, object_hook=hook)

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
            self.logger = logger.bind(id="MAIN")

            for handler in self.config.logging.handlers:
                if handler.enabled:
                    handler.config.enqueue = self.config.parallel.enabled
                    if handler.type == "stderr":
                        handler.config.sink = lambda msg: tqdm.write(msg, end="")
                    if handler.type == "file":
                        handler.config.sink = self.wdir / "simulation.log"
                    self.logger.add(**handler.config, format=formatter)

    def setup_workspace(self):
        if self.config.workspace.overwrite and self.wdir.exists():
            shutil.rmtree(self.wdir)
        self.wdir.mkdir(exist_ok=True)
        if self.config.cache.rebuild and self.cache_path.exists():
            shutil.rmtree(self.cache_path)
        self.cache_path.mkdir(exist_ok=True)

    def reshape_ssc(self):
        ssc = pd.DataFrame.from_records(self.config.platform.ssc, index="month").reindex(np.arange(1, 13, 1))
        ssc = pd.concat([ssc] * 2).reset_index().interpolate(method="cubicspline")
        ssc = ssc.loc[5 : 5 + 11].set_index("month").squeeze().sort_index()
        self.config.platform.ssc = ssc
        self.logger.info("Reshaped and interpolated SSC to monthly frequency.")

    def make_combos(self):
        self.tide_params = utils.make_combos(**self.config.tides.slr, **self.config.tides.amp)
        self.platform_params = utils.make_combos(**self.config.platform)
        self.n = len(self.tide_params) * len(self.platform_params)
        self.logger.info(
            f"Made combination of parameters for a {len(self.tide_params)} different tides and"
            f" {len(self.platform_params)} different platforms for a total of {self.n} runs."
        )

    def configure_parallel(self, mem_per_core):
        self.logger.info(f"Allocating {mem_per_core / 1024 ** 3:.2f} GB of RAM per core.")
        used_mem = psutil.Process(os.getpid()).memory_info().rss
        available_mem = psutil.virtual_memory().available
        self.logger.info(f"RAM - Available: {available_mem / 1024 ** 3:.2f} GB, Used: {used_mem / 1024 ** 3:.2f} GB")
        max_cores = available_mem / mem_per_core
        self.n_cores = int(min(self.config.parallel.max_cores, max_cores))
        self.logger.info(
            f"Max workers set to {self.n_cores} using ~{self.n_cores * mem_per_core / 1024 ** 3:.2f} GB total."
        )

    def build_base(self):
        self.logger.info(f"Loading base tides from pickle file at {self.base_path}.")
        base = pd.read_pickle(self.base_path)
        self.base_size = base.memory_usage()
        self.logger.info(f"Base tide is {self.base_size / 1024 ** 3:.2f} GB.")
        base_pickle = self.cache_path / "base.pickle"
        if not base_pickle.exists():
            self.logger.info(f"Tide object not in cache. Creating from base tides.")
            tide_obj = tides.Tides(water_levels=base)
            self.logger.info(f"Caching base tides.")
            with open(base_pickle, "wb") as file:
                pickle.dump(tide_obj, file, pickle.HIGHEST_PROTOCOL)
        else:
            self.logger.info(f"Tide object already in cache.")

    def build_cache(self):
        def callback(result):
            self.pbar.update()

        freeze_support()
        tqdm.set_lock(RLock())

        self.logger.info(f"Building cache for {len(self.tide_params)} different tides.")
        results = []
        self.pbar = tqdm(desc="TIDES", total=len(self.tide_params), leave=False, unit="tide", dynamic_ncols=True, position=0)
        # pool = mp.Pool(processes=self.n_cores, initializer=init)
        pool = mp.Pool(processes=self.n_cores, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
        for params in self.tide_params:
            kwds = {"cache_dir": self.cache_path, **params}
            results.append(pool.apply_async(func=cache_tide, kwds=kwds, callback=callback))
        pool.close()
        pool.join()
        self.pbar.close()
        self.cached_tides = [r.get() for r in results]
        self.logger.info(f"{sum([r.successful() for r in results])}/{len(results)} tides cached successfully.")

    def write_metadata(self):
        self.metadata = [{**d[0], **d[1]} for d in it.product(self.tide_params, self.platform_params)]
        self.metadata = pd.DataFrame.from_records(self.metadata).drop(columns=["ssc"])
        self.metadata.index.name = "id"
        self.metadata.to_csv(self.wdir / "_metadata.csv")

    def setup(self):
        self.build_base()
        self.configure_parallel(mem_per_core=self.base_size * 7)
        self.build_cache()
        self.write_metadata()
        self.configure_parallel(mem_per_core=self.base_size * 1.1)

    def run(self):
        def callback(result):
            self.pbar.update()

        freeze_support()
        tqdm.set_lock(RLock())

        results = []
        id = 0
        self.pbar = tqdm(desc="MAIN", total=self.n, leave=True, smoothing=0, unit="run", dynamic_ncols=True, position=0)
        # pool = mp.Pool(processes=self.n_cores, initializer=init)
        pool = mp.Pool(processes=self.n_cores, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
        for tide in self.cached_tides:
            for p in self.platform_params:
                params = Bunch(**{k: v for k, v in p.items() if "ssc" not in k})
                params.ssc = p["ssc"] * p["ssc_factor"]
                kwds = {"tide": tide, "wdir": self.wdir, "id": id, **params}
                results.append(pool.apply_async(func=simulate, kwds=kwds, callback=callback))
                id += 1
                time.sleep(0.3)
        pool.close()
        pool.join()
        self.pbar.close()
