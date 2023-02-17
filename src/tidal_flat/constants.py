import pandas as pd

SECOND = pd.Timedelta(1, unit='s')
MINUTE = SECOND * 60
HOUR = MINUTE * 60
DAY = HOUR * 24
WEEK = DAY * 7
MONTH = DAY * 30
YEAR = DAY * 365.2425

CYCLE_PERIOD = 12 * HOUR + 25 * MINUTE
NODAL_PERIOD = DAY * 6798.383

CYCLES_PER_DAY = DAY / CYCLE_PERIOD
CYCLES_PER_YEAR = YEAR / CYCLE_PERIOD

GRAVITY = 9.8  # m/s^2
WATER_DENSITY = 1000.0  # kg/L
WATER_VISCOSITY = 0.001  # Pa s
