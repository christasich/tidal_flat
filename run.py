import argparse
import tidal_marsh as tm
from loguru import logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument('-s', '--start', type=int)
    parser.add_argument('-e', '--end', type=int)
    args = parser.parse_args()
    logger.info(f'Running simulations for RUN# {args.start:04} to {args.end:04}.')
    ids = range(args.start, args.end+1)

    sim = tm.simulation.Simulations(config_path=args.config)
    sim.setup()
    sim.run(ids=ids)
