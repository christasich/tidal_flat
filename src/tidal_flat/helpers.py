import itertools as it
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from loguru import logger
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.utils import Bunch


def interpolate_ts(ts: pd.Series, freq: str = "T") -> pd.Series:
    """Coerces a time series of irregularly spaced data to a regular time series using interpolation.

    Args:
        ts (pd.Series): Irregularly spaced time series.
        freq (str, optional): Frequency string. Defaults to "T".

    Returns:
        pd.Series: Regularly spaced time series.
    """

    norm_index = pd.date_range(ts.index[0].floor(freq), ts.index[-1].ceil(freq), freq=freq)

    ts = pd.concat([ts, pd.DataFrame(columns=ts.columns, index=norm_index)])
    ts = ts[~ts.index.duplicated()].sort_index()
    ts = ts.interpolate(method="time", limit_area="inside")
    return ts.loc[norm_index].dropna()


def make_combos(**kwargs: Any) -> list[Bunch]:
    """Function that takes n-kwargs and returns a list of namedtuples
        for each possible combination of kwargs.

    Returns:
        list[Bunch]: List of bunch objects with each combination of kwargs.
    """

    for key, value in kwargs.items():
        if not isinstance(value, (list, tuple)):
            kwargs[key] = [value]
        if isinstance(value, (np.ndarray)):
            kwargs[key] = value.tolist()
    keys, values = zip(*kwargs.items())
    combos = list(it.product(*values))
    return [Bunch(**dict(zip(keys, combo))) for combo in combos]


def find_pv(data: pd.Series, distance: str | None = None) -> tuple[pd.Series, pd.Series]:
    """Find peaks and valleys in a time series.

    Args:
        data (pd.Series): Time series.
        distance (str | None, optional): Distance between peaks. Defaults to None.

    Returns:
        tuple[pd.Series, pd.Series]: Tuple of series of peaks and valleys.
    """

    if isinstance(distance, (str, pd.Timedelta)):
        step = data.index.to_series().diff().iat[-1]
        distance = pd.to_timedelta(distance) / step

    peaks_iloc = find_peaks(x=data, distance=distance)[0]
    valleys_iloc = find_peaks(x=data * -1, distance=distance)[0]

    return (
        data.iloc[peaks_iloc].rename("elevation").rename_axis("datetime"),
        data.iloc[valleys_iloc].rename("elevation").rename_axis("datetime"),
    )


def linregress_ts(ts: pd.Series, freq: str, ref_date: str | pd.Timestamp) -> Any:
    ref_date = pd.Timestamp(ref_date)
    freq = pd.Timedelta(freq)

    x = ((ts.index - ref_date) / freq).values.reshape(-1, 1)
    y = ts.values.reshape(-1, 1)
    lm = LinearRegression().fit(x, y)

    return (lm, lm.coef_[0, 0], lm.intercept_[0])


def lowess_ts(data: pd.Series, window: pd.Timedelta = None) -> pd.Series:
    endog = data.values
    exog = (data.index - data.index[0]).total_seconds().astype(int).values
    n = data.groupby(by=pd.Grouper(freq=window)).count().mean().round()
    frac = n / len(data)
    y = sm.nonparametric.lowess(endog=endog, exog=exog, frac=frac, is_sorted=True)[:, 1]
    return pd.Series(data=y, index=data.index)


def find_roots(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    roots = np.where(np.diff(np.signbit(a - b)))[0]
    logger.trace(f"{len(roots)} root(s) found.")
    return roots
