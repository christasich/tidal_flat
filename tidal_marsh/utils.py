import itertools as it
import re
from numbers import Number
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from loguru import logger
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.utils import Bunch

from . import constants


def make_combos(**kwargs):
    '''
    Function that takes n-kwargs and returns a list of namedtuples
    for each possible combination of kwargs.
    '''
    for key, value in kwargs.items():
        if isinstance(value, (list, tuple, np.ndarray)) is False:
            kwargs.update({key: [value]})
    keys, values = zip(*kwargs.items())
    combos = [Bunch(n=i, **dict(zip(keys, combo))) for i, combo in enumerate(it.product(*values))]
    return combos


def construct_filename(fn_format, **kwargs):
    '''
    Function that takes a string with n-number of format placeholders (e.g. {0]})
    and uses the values from n-kwargs to populate the string.
    '''
    kwarg_num = len(kwargs)
    fn_var_num = len(re.findall(r'\{.*?\}', fn_format))
    if kwarg_num != fn_var_num:
        raise Exception(
            'Format error: Given {} kwargs, but '
            'filename format has {} sets of '
            'braces.'.format(kwarg_num, fn_var_num)
        )
    fn = fn_format.format(*kwargs.values())
    return fn


def search_file(wdir, filename):
    '''
    Function that searches a directory for a filename and returns the number
    of exact matches (0 or 1). If more than one file is found, the function
    will raise an exception.
    '''
    if len(list(Path(wdir).glob(filename))) == 0:
        found = 0
    elif len(list(Path(wdir).glob(filename))) == 1:
        found = 1
    elif len(list(Path(wdir).glob(filename))) > 1:
        raise Exception('Found too many files that match.')
    return found


def find_pv(data: pd.Series, window: str):

    distance = pd.Timedelta(window) / pd.Timedelta(data.index.freq)

    peaks_iloc = find_peaks(x=data, distance=distance)[0]
    valleys_iloc = find_peaks(x=data * -1, distance=distance)[0]

    return (data.iloc[peaks_iloc], data.iloc[valleys_iloc])


def regress_ts(ts: pd.Series, freq: str, ref_date: str | pd.Timestamp):
    ref_date = pd.Timestamp(ref_date)
    freq = pd.Timedelta(freq)

    x = ((ts.index - ref_date) / freq).values.reshape(-1, 1)
    y = ts.values.reshape(-1, 1)
    lm = LinearRegression().fit(x, y)

    return (lm, lm.coef_[0, 0], lm.intercept_[0])


def lowess_ts(data: pd.Series, window: pd.Timedelta = None):
    endog = data.values
    exog = (data.index - data.index[0]).total_seconds().astype(int).values
    n = data.groupby(by=pd.Grouper(freq=window)).count().mean().round()
    frac = n / len(data)
    y = sm.nonparametric.lowess(endog=endog, exog=exog, frac=frac, is_sorted=True)[:, 1]
    return pd.Series(data=y, index=data.index)


def make_trend(
    rate: Number | pd.Series,
    time_unit: str,
    index: pd.DatetimeIndex,
) -> pd.Series:
    rate = rate / (pd.Timedelta(time_unit) / pd.Timedelta(index.freq))
    if isinstance(rate, Number):
        trend = pd.Series(data=rate, index=index).cumsum()
    elif isinstance(rate, pd.Series):
        trend = rate.reindex(index).interpolate().cumsum()
    return trend


from typing import Callable, ParamSpec, TypeVar

T = TypeVar('T')
P = ParamSpec('P')


def datetime2num(t: pd.Timestamp) -> float | np.ndarray:
    try:
        return t.timestamp()
    except AttributeError:
        pass
    try:
        return (t.astype(np.int64) / 10**9).values
    except:
        raise ValueError('t must be a pd.Timestamp or pd.DatetimeIndex.')


def num2datetime(t: float | int) -> pd.Timestamp | pd.DatetimeIndex:
    try:
        return pd.to_datetime(t, unit='s')
    except:
        raise ValueError('t must be a number.')


def datetime_wrapper(fun: Callable[P, T]) -> Callable[P, T]:
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        args = [datetime2num(arg) if isinstance(arg, (pd.Timestamp, pd.DatetimeIndex)) else arg for arg in args]
        kwargs = dict(
            (key, datetime2num(value)) if isinstance(value, (pd.Timestamp, pd.DatetimeIndex)) else (key, value)
            for key, value in kwargs.items()
        )
        return fun(*args, **kwargs)

    return wrapped


def stokes_settling(
    grain_diameter: float,
    grain_density: float,
    fluid_density: float = constants.WATER_DENSITY,
    fluid_viscosity: float = constants.WATER_VISCOSITY,
    gravity: float = constants.GRAVITY,
) -> float:
    settling_rate = (2 / 9 * (grain_density - fluid_density) / fluid_viscosity) * gravity * (grain_diameter / 2) ** 2
    return settling_rate


def find_roots(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    roots = np.where(np.diff(np.signbit(a - b)))[0]
    return roots


def calculate_rate_vector(
    t: pd.Timestamp | pd.DatetimeIndex, rate: float, ref_t: pd.Timedelta = None
) -> float | np.ndarray:
    if ref_t is None:
        ref_t = t[0]
    elapsed_seconds = (t - ref_t).total_seconds()
    if isinstance(t, pd.DatetimeIndex):
        return (rate * elapsed_seconds).values
    else:
        return rate * elapsed_seconds


def load_config(config):
    with open(config) as file:
        config = yaml.safe_load(file)
    return config
