import multiprocessing as mp
from multiprocessing.pool import Pool
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
from joblib import Parallel, delayed, dump, load
from loguru import _Logger, logger
from tqdm.auto import tqdm
from types import SimpleNamespace
import os
import itertools as it

from . import tides
from . import utils
from .core import Model
from .tides import Tides


def simulate(params):

    _logger = logger.bind(id=f"{params.id:04}")

    _logger.info("Initializing sediment model.")
    model = Model(
        water_levels=params.water_levels,
        initial_elevation=params.initial_elevation,
        ssc=params.ssc,
        grain_diameter=params.grain_diameter,
        grain_density=params.grain_density,
        bulk_density=params.bulk_density,
        organic_rate=params.organic_rate,
        compaction_rate=params.compaction_rate,
        subsidence_rate=params.subsidence_rate,
        id=params.id,
    )
    try:
        _logger.info("Starting simulation.")
        model.run()

        result_path = params.wdir / "results" / f"{params.id:04}.csv"
        model.results.to_csv(result_path)
        logger.info(f"Model completed successfully. Saved results to {result_path}")
        with open(params.wdir / "overextraction.csv", "a") as file:
            writer = csv.writer(file)
            writer.writerow([f"{params.id:04}", model.overextraction])
    except:
        _logger.exception("Issue running model.")


@dataclass
class Simulations:
    config_path: InitVar[str]
    config: Bunch = field(init=False)
    wdir: Path = field(init=False)
    logger: _Logger = field(init=False)
    tide: Tides = field(init=False)
    tide_params: Bunch = field(init=False)
    platform_params: Bunch = field(init=False)
    n: int = field(init=False)
    metadata: pd.DataFrame = field(init=False)
    pbar: tqdm = field(init=False, repr=False)
    results: list = field(init=False, default_factory=list)

    def __post_init__(self, config_path):
        # Load configuration
        self.config = utils.load_config(config_path)

        # Setup working directory
        self.wdir = Path(self.config.wdir)
        if self.wdir.exists():
            shutil.rmtree(self.wdir)
        self.wdir.mkdir()

        if self.config.logging.enabled:
            self.configure_logger()

        self.load_tides()
        self.reshape_ssc()
        self.make_combos()
        self.pbar = tqdm(desc="MAIN", total=self.n, leave=True, unit="run")
        self.write_metadata()

    def configure_logger(self):
        logger.enable("tidal_marsh")
        logger.remove()
        self.logger = logger.bind(id="MAIN")

        for handler in self.config.logging.handlers:
            if handler.enabled:
                handler.config.enqueue = self.config.parallel.enabled
                if handler.type == "stderr":
                    handler.config.sink = lambda msg: tqdm.write(msg, end="")
                if handler.type == "file":
                    handler.config.sink = self.wdir / "simulations.log"
                self.logger.add(**handler.config)

    def load_tides(self):
        feather_path = Path(self.config.tides.feather_path)
        cache_path = Path(self.config.tides.cache_path)
        pickle_path = Path(cache_path / (feather_path.stem + ".obj"))
        if self.config.tides.rebuild or not pickle_path.exists():
            self.logger.info(f"Loading tides from feather file at {feather_path}")
            data = tides.load_tides(feather_path.as_posix())

            self.logger.info("Building Tides object")
            self.tide = tides.Tides(water_levels=data)

            self.logger.info(f"Caching tides to {pickle_path}.")
            with open(pickle_path, "wb") as file:
                pickle.dump(self.tide, file, pickle.HIGHEST_PROTOCOL)
        else:
            self.logger.info("Loading tides from cache.")
            with open(pickle_path, "rb") as file:
                self.tide = pickle.load(file)
        self.logger.info("Tides loaded successfully.")

    def reshape_ssc(self):
        ssc = pd.DataFrame.from_records(self.config.platform.parameters.ssc, index="month").reindex(np.arange(1, 13, 1))
        ssc = pd.concat([ssc] * 2).reset_index().interpolate(method="cubicspline")
        ssc = ssc.loc[5 : 5 + 11].set_index("month").squeeze().sort_index()
        self.config.platform.parameters.ssc = ssc
        self.logger.info("Reshaped and interpolated SSC to monthly frequency.")

    def make_combos(self):
        self.tide_params = utils.make_combos(**self.config.tides.parameters)
        self.platform_params = utils.make_combos(**self.config.platform.parameters)
        self.n = len(self.tide_params) * len(self.platform_params)
        self.logger.info(f"Made combination of parameters for a total of {self.n} runs.")

    def write_metadata(self):
        self.metadata = [{**d[0], **d[1]} for d in it.product(self.tide_params, self.platform_params)]
        self.metadata = pd.DataFrame.from_records(self.metadata).drop(columns=["ssc"])
        self.metadata.index.name = "id"
        self.metadata.to_csv(self.wdir / "metadata.csv")

    def prep_tides(self, z2100, b, beta, k):
        self.logger.info(f"Preparing tides with parameters: z2100={z2100}, b={b}, beta={beta}, k={k}.")
        water_levels = (
            self.tide.data.elevation
            + self.tide.calc_slr(z2100=z2100, b=b)
            + self.tide.calc_amplification(beta=beta, k=k)
        )
        return water_levels

    def single(self):
        id = 0
        for t in self.tide_params:
            water_levels = self.prep_tides(**t)
            for p in self.platform_params:
                params = Bunch(**{k: v for k, v in p.items() if "ssc" not in k})
                params.ssc = p["ssc"] * p["ssc_factor"]
                params.water_levels = water_levels
                params.id = id
                params.wdir = self.wdir
                result = simulate(params)
                self.results.append(result)
                id += 1
        self.pbar.close()
        self.logger.info("All simulations complete.")

    def multiple(self):
        def init():
            sys.stdout.write(" ")
            sys.stdout.flush()

        def callback(*a):
            self.pbar.update()

        max_cores = psutil.virtual_memory().total / (self.tide.data.memory_usage().sum() * 1.25)
        self.config.parallel.pool.processes = int(min(self.config.parallel.max_cores, max_cores))
        self.logger.info(f"Initializing multiprocessing with a pool of {self.config.parallel.pool.processes} workers.")
        pool = mp.Pool(initializer=init, **self.config.parallel.pool)

        id = 0
        count = 1
        for t in self.tide_params:
            batch_results = []
            water_levels = self.prep_tides(**t)
            self.logger.info(f"Starting batch #{count}/{len(self.tide_params)}.")
            for p in self.platform_params:
                params = Bunch(**{k: v for k, v in p.items() if "ssc" not in k})
                params.ssc = p["ssc"] * p["ssc_factor"]
                params.water_levels = water_levels
                params.id = id
                params.wdir = self.wdir
                result = pool.apply_async(func=simulate, args=(params,), callback=callback)
                batch_results.append(result)
                self.results.append(result)
                self.logger.debug(f"RUN #{id:04} sent to worker.")
                id += 1
            self.logger.info(f"Batch #{count} sent to workers. Waiting for results.")
            [r.wait() for r in batch_results]
            self.logger.info(f"Batch #{count} complete.")
            count += 1
        pool.close()
        pool.join()
        self.pbar.close()
        self.logger.info("All batches complete.")

    def run(self):
        if self.config.parallel.enabled:
            self.multiple()
        else:
            self.single()


#         # def run(self):

#         #     if self.config.parallel.enabled:

#         #         def init():
#         #             sys.stdout.write(" ")
#         #             sys.stdout.flush()

#         #         def callback(*a):
#         #             self.pbar.update()

#         #         max_cores = psutil.virtual_memory().total / (self.tide.data.memory_usage().sum() * 1.25)
#         #         self.config.parallel.pool.processes = int(min(self.config.parallel.max_cores, max_cores))
#         #         self.pool = mp.Pool(initializer=init, **self.config.parallel.pool)
#         #         self.logger.info(
#         #             f"Starting parallel processing with pool of {self.config.parallel.pool.processes} workers."
#         #         )

#         #     id = 0
#         #     for t in self.tide_params:
#         #         self.logger.info(f"Preparing tides for z2100={t.z2100}, b={t.b}, beta={t.beta}, k={t.k}.")
#         #         water_levels = (
#         #             self.tide.data.elevation
#         #             + self.tide.calc_slr(z2100=t.z2100, b=t.b)
#         #             + self.tide.calc_amplification(beta=t.beta, k=t.k)
#         #         )
#         #         self.logger.info(f"Starting runs for z2100={t.z2100}, b={t.b}, beta={t.beta}, k={t.k}.")
#         #         params = Bunch(water_levels=water_levels, id=id, platform_params=self.platform_params)
#         #         if self.config.parallel.enabled:
#         #             self.pool.imap_unordered(func=self.run_one, iterable=params)
#         #         for p in self.platform_params:
#         #             params = {k: v for k, v in p.items() if "ssc" not in k}
#         #             params.update(ssc=p["ssc"] * p["ssc_factor"])
#         #             if self.config.parallel.enabled:
#         #                 result = self.pool.apply_async(
#         #                     func=self.run_one, args=(water_levels, params, id), callback=callback
#         #                 )
#         #                 self.logger.info(f"RUN #{id:04} sent to worker.")
#         #                 # time.sleep(15)
#         #             else:
#         #                 result = self.run_one(water_levels, params, id)
#         #             self.results.append(result)
#         #             id += 1
#         #     if self.config.parallel.enabled:
#         #         self.pool.close()
#         #         self.logger.info("All jobs sent. Waiting on results.")
#         #         [r.wait() for r in self.results]
#         #         self.logger.info("All results received.")
#         #     self.pbar.close()
#         # self.logger.info("All simulations complete.")


# @logger.catch
# def run(id, params, water_levels):

#     with logger.contextualize(run=f"{id:04}"):

#         ssc = params.ssc.values * params.ssc_factor

#         logger.info("Initializing sediment model")
#         model = Model(
#             water_levels=water_levels,
#             initial_elevation=params.elevation_start,
#             ssc=ssc,
#             grain_diameter=params.grain_diameter,
#             grain_density=params.grain_density,
#             bulk_density=params.bulk_density,
#             organic_rate=params.organic_rate,
#             compaction_rate=params.compaction_rate,
#             subsidence_rate=params.subsidence_rate,
#             id=id,
#         )
#         try:
#             logger.info("Starting simulation")
#             model.run()
#         except:
#             logger.exception("Issue running model")
#         else:
#             filename = f"{model.id:04}.csv"
#             result_path = params.wdir / filename
#             logger.info(f"Model completed successfully. Saved results to {result_path}")
#             model.results.to_csv(result_path)
#             with open(params.wdir / "overextraction.csv", "a") as file:
#                 writer = csv.writer(file)
#                 writer.writerow([f"{model.id:04}", model.overextraction])


# @logger.contextualize(run="MAIN")
# def run_all(config_file):

#     config = utils.load_config(config_file)

#     wdir = Path(config.wdir)
#     if wdir.exists():
#         shutil.rmtree(wdir)
#     wdir.mkdir()

#     def formatter(record):
#         if "model_time" in record["extra"]:
#             end = " | <yellow>{extra[model_time]}</yellow> | <level>{message}</level>\n{exception}"
#         else:
#             end = " | <level>{message}</level>\n{exception}"
#         return config.logging.format + end

#     if config.logging.enabled:
#         logger.enable("tidal_marsh")
#         logger.remove()
#         for handler in config.logging.handlers:
#             if handler.enabled:
#                 handler.config.enqueue = config.parallel.n_jobs != 1
#                 if handler.type == "stderr":
#                     handler.config.sink = lambda msg: tqdm.write(msg, end="")
#                 if handler.type == "file":
#                     handler.config.sink = wdir / "_parallel.log"
#                 logger.add(**handler.config, format=formatter)

#     logger.info("Beginning main process")
#     feather_path = Path(config.tides.feather_file)
#     cache_dir = Path(config.tides.cache_dir)
#     pickle_path = Path(cache_dir / (feather_path.stem + ".obj"))

#     if config.tides.rebuild or not pickle_path.exists():
#         logger.info(f"Loading tides from feather file at {feather_path}")
#         data = tides.load_tides(feather_path.as_posix())

#         logger.info("Building Tides object")
#         tide = tides.Tides(water_levels=data)

#         logger.info("Pickling tides")
#         with open(pickle_path, "wb") as file:
#             pickle.dump(tide, file, pickle.HIGHEST_PROTOCOL)
#     else:
#         logger.info("Loading tides from cache.")
#         with open(pickle_path, "rb") as file:
#             tide = pickle.load(file)

#     memory_limit = tide.data.memory_usage(deep=True).sum() * 2

#     ssc = pd.DataFrame.from_records(config.parameters.ssc, index="month").reindex(np.arange(1, 13, 1))
#     ssc = pd.concat([ssc] * 2).reset_index().interpolate(method="cubicspline")
#     ssc = ssc.loc[5 : 5 + 11].set_index("month").squeeze().sort_index()
#     config.parameters.ssc = ssc

#     config.parameters.wdir = wdir

#     logger.info("Making combos of parameters.")
#     param_combos = utils.make_combos(**config.parameters)
#     tide_combos = utils.make_combos(**config.tides.parameters)
#     n = len(tide_combos) * len(param_combos)

#     info = [{**d[0], **d[1]} for d in it.product(tide_combos, param_combos)]
#     info = pd.DataFrame.from_records(info).drop(columns=["ssc", "wdir"])
#     info.index.name = "id"
#     info.to_csv(wdir / "_info.csv")

#     def init():
#         sys.stdout.write(" ")
#         sys.stdout.flush()

#     def callback(*a):
#         pbar.update()

#     logger.info(f"Starting parallel processing with pool of {config.parallel.n_jobs} workers.")
#     pbar = tqdm(desc="MAIN", total=n, leave=True, unit="run")
#     id = 0
#     with mp.Pool(processes=config.parallel.n_jobs, initializer=init, maxtasksperchild=1) as pool:
#         results = []
#         for t in tide_combos:
#             logger.info(f"Preparing tides for z2100={t.z2100}, b={t.b}, beta={t.beta}, k={t.k}.")
#             water_levels = (
#                 tide.data.elevation + tide.calc_slr(z2100=t.z2100, b=t.b) + tide.calc_amplification(beta=t.beta, k=t.k)
#             )
#             logger.info(f"Starting runs for z2100={t.z2100}, b={t.b}, beta={t.beta}, k={t.k}.")
#             for p in param_combos:
#                 check_system_resources(id=id, memory_limit=memory_limit)
#                 results.append(pool.apply_async(func=run, args=(id, p, water_levels), callback=callback))
#                 time.sleep(15)
#                 id += 1
#         logger.info("All jobs sent. Waiting on results.")
#         [r.wait() for r in results]
#         logger.info("All results received.")
#         pbar.close()
#     logger.info("All simulations complete.")


# def check_system_resources(id, memory_limit):
#     used_frac = psutil.virtual_memory().used / psutil.virtual_memory().total
#     while psutil.virtual_memory().available < memory_limit:
#         logger.info(f"Waiting for more free memory before initializing RUN #{id:04}.")
#         logger.info(f"Used RAM: {psutil.virtual_memory().used / 1024 ** 3:.2f} GB; Percent: {used_frac:.2%}")
#         time.sleep(5)
#     logger.info(f"Sending RUN #{id:04} to worker.")
#     logger.info(f"Used RAM: {psutil.virtual_memory().used / 1024 ** 3:.2f} GB; Percent: {used_frac:.2%}")
