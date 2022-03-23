import tidal_marsh as tm

sim = tm.simulations.Simulations(config_path="config.yaml")
sim.tide_params = sim.tide_params[4:6]
sim.configure_parallel()
sim.build_cache()
