import tidal_marsh as tm

config_path = "config.yaml"

sim = tm.Simulations(config_path=config_path)
sim.setup()
sim.run()
