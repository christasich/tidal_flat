# import pathlib

# ROOT = pathlib.Path(__file__).resolve().parents[1]

import pandas as pd

SECOND = 1
MINUTE = SECOND * 60
HOUR = MINUTE * 60
DAY = HOUR * 24
WEEK = DAY * 7
MONTH = DAY * 30
YEAR = DAY * 365
GRAVITY = 9.8  # m/s^2
WATER_DENSITY = 1000.0  # kg/L
WATER_VISCOSITY = 0.001  # Pa s

TIDAL_PERIOD = pd.Timedelta("12H25T")
NODAL_PERIOD = pd.Timedelta("365.25D") * 18.61
