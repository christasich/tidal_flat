from collections import namedtuple
import inspect
import utide
from matplotlib import dates as mdates
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
import itertools as it
from pyarrow import feather


def make_combos(**kwargs):
    """
    Function that takes n-kwargs and returns a list of namedtuples
    for each possible combination of kwargs.
    """
    for key, value in kwargs.items():
        if isinstance(value, (list, tuple, np.ndarray)) is False:
            kwargs.update({key: [value]})
    keys, value_tuples = zip(*kwargs.items())
    combo_tuple = namedtuple("combos", ["n", *list(keys)])
    combos = [combo_tuple(n, *values) for n, values in enumerate(it.product(*value_tuples))]
    return combos


def construct_filename(fn_format, **kwargs):
    """
    Function that takes a string with n-number of format placeholders (e.g. {0]})
    and uses the values from n-kwargs to populate the string.
    """
    kwarg_num = len(kwargs)
    fn_var_num = len(re.findall(r"\{.*?\}", fn_format))
    if kwarg_num != fn_var_num:
        raise Exception(
            "Format error: Given {0} kwargs, but "
            "filename format has {1} sets of "
            "braces.".format(kwarg_num, fn_var_num)
        )
    fn = fn_format.format(*kwargs.values())
    return fn


def search_file(wdir, filename):
    """
    Function that searches a directory for a filename and returns the number
    of exact matches (0 or 1). If more than one file is found, the function
    will raise an exception.
    """
    if len(list(Path(wdir).glob(filename))) == 0:
        found = 0
    elif len(list(Path(wdir).glob(filename))) == 1:
        found = 1
    elif len(list(Path(wdir).glob(filename))) > 1:
        raise Exception("Found too many files that match.")
    return found


def missing_combos(wdir, fn_format, combos):
    """
    Function that creates filenames for a list of combinations and 
    then searches a directory for the filenames. The function returns
    a list of combinations that were not found.
    """
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


def make_tides(coef: utide._solve.Bunch, start:int, stop:int, freq: str, n_jobs=1, pbar=False):

    years = np.arange(start, stop, 1)
    if pbar==True:
        years = tqdm(years)

    def one_year(year, coef, freq):
        start = str(year)
        end = str(year+1)
        index = pd.date_range(start=start, end=end, closed="left", freq=freq, name="datetime")
        time = mdates.date2num((index - pd.Timedelta("6 hours")).to_pydatetime())
        elev = utide.reconstruct(t=time, coef=coef, verbose=False).h
        
        return(pd.Series(data=elev, index=index))

    tides = Parallel(n_jobs=n_jobs)(delayed(one_year)(year=year, coef=coef, freq=freq) for year in years)
    
    return(pd.concat(tides))


def load_tide(wdir, filename):
    """
    Function that loads the tidal curve constructed by make_tides.R
    and sets the index to the Datetime column and infers frequency.
    """
    fp = wdir / filename
    data = pd.read_feather(fp)
    vals = data.elevation.values
    index = pd.DatetimeIndex(data.datetime, freq="infer")
    tides = pd.Series(data=vals, index=index, name="elevation")

    return tides


def make_param_tuple(
    water_height,
    index,
    bound_conc,
    settle_rate,
    bulk_den,
    start_elev=0,
    tidal_amplifier=1,
    conc_method="CT",
    organic_rate=0,
    compaction_rate=0,
    subsidence_rate=0,
):
    param_tuple = namedtuple(
        "param_tuple", inspect.getfullargspec(make_param_tuple).args
    )
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
