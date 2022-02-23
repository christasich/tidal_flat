import multiprocessing as mp
import os
import pickle
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from joblib import Parallel, delayed, dump, load
from loguru import logger
from tqdm.auto import tqdm

from . import tides, utils
from .core import Model


@logger.catch
def run(params):

    with logger.contextualize(run=f"{params.id:04}"):

        logger.info(f"Beginning run {params.id}")

        # while psutil.virtual_memory().free / psutil.virtual_memory().total < 0.5:
        #     logger.info(
        #         f"Waiting for more free memory before loading tides. {psutil.virtual_memory().used / psutil.virtual_memory().total:.2%} used"
        #     )
        #     time.sleep(5)

        logger.info("Loading pickled tides")
        with open(params.pickle_path, "rb") as file:
            tide = pickle.load(file)

        logger.info("Adjusting tides for slr and beta")
        water_levels = tide.modify(beta=params.beta, benchmarks=["MHW", "MLW"], trend=params.slr_rate)
        del tide

        ssc = params.ssc.values * params.ssc_factor

        logger.info("Initializing model")
        model = Model(
            water_levels=water_levels,
            initial_elevation=params.elevation_start,
            ssc=ssc,
            grain_diameter=params.grain_diameter,
            grain_density=params.grain_density,
            bulk_density=params.bulk_density,
            organic_rate=params.organic_rate,
            compaction_rate=params.compaction_rate,
            subsidence_rate=params.subsidence_rate,
            id=params.id,
        )
        try:
            logger.info("Starting simulation")
            model.run()
        except:
            logger.exception("Issue running model")
        else:
            filename = f"{model.id:04}.csv"
            result_path = params.wdir / filename
            model.results.to_csv(result_path)
            logger.info(f"Model completed successfully. Saved results to {result_path}")


def run_all(config_file):
    config = utils.load_config(config_file)

    wdir = Path(config["wdir"])
    if wdir.exists():
        shutil.rmtree(wdir)
    wdir.mkdir()

    if config["logging"]["enabled"]:
        logger.enable("tidal_marsh")
        logger.remove()
        for handler in config["logging"]["handlers"]:
            if config["logging"]["handlers"][handler]["enabled"]:
                handler_config = config["logging"]["handlers"][handler]["config"]
                handler_config["enqueue"] = config["parallel"]["n_jobs"] != 1
                if handler == "stderr":
                    handler_config["sink"] = lambda msg: tqdm.write(msg, end="")
                if handler == "file":
                    handler_config["sink"] = wdir / handler_config["sink"]
                logger.add(**handler_config)

    with logger.contextualize(run="MAIN"):
        logger.info("Beginning main process")
        feather_path = Path(config["tides"]["feather_file"])
        cache_dir = Path(config["tides"]["cache_dir"])
        pickle_path = Path(cache_dir / (feather_path.stem + ".obj"))

        if config["tides"]["rebuild"] or not pickle_path.exists():
            logger.info(f"Loading tides from feather file at {config['tides']['feather_file']}")
            data = tides.load_tides(config["tides"]["feather_file"])

            logger.info("Building Tides object")
            tide = tides.Tides(water_levels=data)

            logger.info("Pickling tides")
            with open(pickle_path, "wb") as file:
                pickle.dump(tide, file, pickle.HIGHEST_PROTOCOL)

        ssc = pd.DataFrame.from_records(config["parameters"]["ssc"], index="month").reindex(np.arange(1, 13, 1))
        ssc = pd.concat([ssc] * 2).reset_index().interpolate(method="cubicspline")
        ssc = ssc.loc[5 : 5 + 11].set_index("month").squeeze().sort_index()
        config["parameters"]["ssc"] = ssc

        config["parameters"]["pickle_path"] = pickle_path
        config["parameters"]["wdir"] = wdir

        logger.info("Making combos of parameters")
        combos = utils.make_combos(**config["parameters"])
        logger.debug("Shuffling combos for better runtime prediction.")
        random.shuffle(combos)

        logger.info(f"Writing model info csv to {wdir}/_info.csv")
        info = pd.DataFrame.from_records(combos).drop(columns="ssc")
        info.id = info.id.apply(lambda id: f"{id:04}")
        info.set_index("id").to_csv(wdir / "_info.csv")

        logger.info("===BEGINNING SIMULATIONS===")
        with mp.Pool(processes=config["parallel"]["n_jobs"], initializer=display_hack) as pool:
            for _ in tqdm(pool.imap(run, combos), desc="MAIN", total=len(combos), leave=True, unit="run"):
                pass
        logger.info("===SIMULATIONS COMPLETED===")


def display_hack():
    sys.stdout.write(" ")
    sys.stdout.flush()
