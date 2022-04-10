import argparse
import tidal_marsh as tm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config')
    parser.add_argument('-s', '--start', type=int)
    parser.add_argument('-e', '--end', type=int)
    args = parser.parse_args()

    sim = tm.simulation.Simulations(config_path=args.config)

    if not args.end:
        args.end = sim.metadata.index[-1]

    ids = range(args.start, args.end+1)
    sim.setup()
    sim.run(ids=ids)
