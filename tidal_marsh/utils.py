import inspect
import itertools as it
import re
from collections import namedtuple
from numbers import Number
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import utide
from joblib import Parallel, delayed
from loguru import logger
from matplotlib import dates as mdates
from pyarrow import feather
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from tqdm.notebook import tqdm

from . import constants


def make_combos(**kwargs):
    '''
    Function that takes n-kwargs and returns a list of namedtuples
    for each possible combination of kwargs.
    '''
    for key, value in kwargs.items():
        if isinstance(value, (list, tuple, np.ndarray)) is False:
            kwargs.update({key: [value]})
    keys, value_tuples = zip(*kwargs.items())
    combo_tuple = namedtuple('combos', ['n', *list(keys)])
    combos = [combo_tuple(n, *values) for n, values in enumerate(it.product(*value_tuples))]
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


def missing_combos(wdir, fn_format, combos):
    '''
    Function that creates filenames for a list of combinations and
    then searches a directory for the filenames. The function returns
    a list of combinations that were not found.
    '''
    to_make = []
    for combo in combos:
        fn = construct_filename(
            fn_format=fn_format,
            run_len=combo.run_len,
            dt=int(pd.to_timedelta(combo.dt).total_seconds()),
            slr=combo.slr,
        )
        if search_file(wdir, fn) == 0:
            to_make.append(combo)
    return to_make


def make_tides(coef: utide._solve.Bunch, start: int, stop: int, freq: str, n_jobs=1, pbar=False):

    years = np.arange(start, stop, 1)
    if pbar == True:
        years = tqdm(years)

    def one_year(year, coef, freq):
        start = str(year)
        end = str(year + 1)
        index = pd.date_range(start=start, end=end, closed='left', freq=freq, name='datetime')
        time = mdates.date2num((index - pd.Timedelta('6 hours')).to_pydatetime())
        elev = utide.reconstruct(t=time, coef=coef, verbose=False).h

        return pd.Series(data=elev, index=index)

    tides = Parallel(n_jobs=n_jobs)(delayed(one_year)(year=year, coef=coef, freq=freq) for year in years)

    return pd.concat(tides)


def make_param_tuple(
    water_height,
    index,
    bound_conc,
    settle_rate,
    bulk_den,
    start_elev=0,
    tidal_amplifier=1,
    conc_method='CT',
    organic_rate=0,
    compaction_rate=0,
    subsidence_rate=0,
):
    param_tuple = namedtuple('param_tuple', inspect.getfullargspec(make_param_tuple).args)
    params = param_tuple(
        water_height=water_height,
        index=index,
        bound_conc=bound_conc,
        settle_rate=settle_rate,
        bulk_den=bulk_den,
        start_elev=start_elev,
        tidal_amplifier=tidal_amplifier,
        conc_method=conc_method,
        organic_rate=organic_rate,
        compaction_rate=compaction_rate,
        subsidence_rate=subsidence_rate,
    )
    return params


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
