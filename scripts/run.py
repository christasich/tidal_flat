import tidal_marsh as tm
from sklearn.utils import Bunch

config_path = "./config.yaml"

sim = tm.Simulations(config_path=config_path)
sim.prepare_tides()
sim.run()
