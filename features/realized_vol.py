"""
Realized volatility estimators for the Neural Volatility Surface Forecasting System.

Provides multiple estimators:
- Close-to-close rolling RV
- Parkinson high-low estimator
- Yang-Zhang OHLC estimator
- GARCH(1,1) forecast via arch library
- IV-RV spread
"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    warnings.warn("arch library not available; garch_vol_forecast will raise ImportError.")

ANNUALIZATION_FACTOR = np.sqrt(252)


def _rolling_window(arr: np.ndarray, window: int) -> np.ndarray:
    """Create a rolling window view of a 1-D array."""
    if window > len(arr):
        raise ValueError(f"window ({window}) exceeds array length ({len(arr)})")
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def compute_realized_vol(
    returns: np.ndarray,
    window: int,
    annualize: bool = True,
) -> np.ndarray:
    """
    Rolling close-to-close realized volatility.

    Parameters
    ----------
    returns : np.ndarray, shape (N,)
        Log or simple daily returns.
    window : int
        Rolling window in trading days.
    annualize : bool
        Multiply by sqrt(252) when True.

    Returns
    -------
    np.ndarray, shape (N,)
        Rolling realized volatility; first (window-1) entries are NaN.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    out = np.full(n, np.nan)

    if window < 2:
        raise ValueError("window must be >= 2")

    windows = _rolling_window(returns, window)          # (n - window + 1, window)
    rv = np.std(windows, axis=1, ddof=1)
    if annualize:
        rv = rv * ANNUALIZATION_FACTOR

    out[window - 1:] = rv
    return out


def parkinson_vol(
    high: np.ndarray,
    low: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Parkinson (1980) high-low volatility estimator.

    Formula: sqrt( 1/(4*ln2) * mean(ln(H/L)^2) * 252 )

    Parameters
    ----------
    high, low : np.ndarray, shape (N,)
        Daily high and low prices (must be positive).
    window : int
        Rolling window in trading days.

    Returns
    -------
    np.ndarray, shape (N,)
        Annualized Parkinson volatility; first (window-1) entries are NaN.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)

    if np.any(low <= 0):
        raise ValueError("low prices must be positive")
    if np.any(high < low):
        raise ValueError("high must be >= low for all entries")

    log_hl_sq = np.log(high / low) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))

    n = len(high)
    out = np.full(n, np.nan)
    windows = _rolling_window(log_hl_sq, window)        # (n - window + 1, window)
    park = np.sqrt(factor * np.mean(windows, axis=1) * 252.0)
    out[window - 1:] = park
    return out


def yang_zhang_vol(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Yang-Zhang (2000) OHLC volatility estimator.

    Combines overnight (close-to-open) variance, open-to-close variance, and
    the Rogers-Satchell estimator via an optimal weighting coefficient k.

    Formula:
        sigma^2 = sigma_overnight^2 + k * sigma_oc^2 + (1-k) * sigma_rs^2
        k = 0.34 / (1.34 + (window+1)/(window-1))

    Parameters
    ----------
    open_, high, low, close : np.ndarray, shape (N,)
        OHLC price series.
    window : int
        Rolling window in trading days.

    Returns
    -------
    np.ndarray, shape (N,)
        Annualized Yang-Zhang volatility; first (window-1) entries are NaN.
    """
    open_ = np.asarray(open_, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)

    n = len(close)
    if not (len(open_) == len(high) == len(low) == n):
        raise ValueError("All OHLC arrays must have the same length")

    # Overnight return: log(Open_t / Close_{t-1})
    overnight = np.full(n, np.nan)
    overnight[1:] = np.log(open_[1:] / close[:-1])

    # Open-to-close return: log(Close_t / Open_t)
    oc = np.log(close / open_)

    # Rogers-Satchell component per day
    rs = (
        np.log(high / close) * np.log(high / open_)
        + np.log(low / close) * np.log(low / open_)
    )

    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))
    out = np.full(n, np.nan)

    for i in range(window, n + 1):
        on_w = overnight[i - window: i]
        oc_w = oc[i - window: i]
        rs_w = rs[i - window: i]

        # Drop the first overnight which is NaN if i == window and window > 1
        valid_on = on_w[~np.isnan(on_w)]
        if len(valid_on) < 2:
            continue

        mu_on = np.mean(valid_on)
        mu_oc = np.mean(oc_w)

        var_on = np.sum((valid_on - mu_on) ** 2) / (len(valid_on) - 1)
        var_oc = np.sum((oc_w - mu_oc) ** 2) / (window - 1)
        var_rs = np.mean(rs_w)

        var_yz = var_on + k * var_oc + (1.0 - k) * var_rs
        out[i - 1] = np.sqrt(max(var_yz, 0.0) * 252.0)

    return out


def garch_vol_forecast(
    returns: np.ndarray,
    horizon: int = 5,
) -> float:
    """
    Fit a GARCH(1,1) model and return the h-step ahead annualized volatility forecast.

    Parameters
    ----------
    returns : np.ndarray, shape (N,)
        Percentage or log returns (will be scaled to percentage internally).
    horizon : int
        Number of steps ahead to forecast.

    Returns
    -------
    float
        Annualized volatility forecast for h steps ahead.

    Raises
    ------
    ImportError
        If the arch library is not installed.
    """
    if not ARCH_AVAILABLE:
        raise ImportError(
            "arch library is required for GARCH forecasting. "
            "Install with: pip install arch"
        )

    returns = np.asarray(returns, dtype=float)

    # arch expects percentage returns for numerical stability
    scale = 100.0 if np.max(np.abs(returns)) < 1.0 else 1.0
    pct_returns = returns * scale

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(pct_returns, vol="Garch", p=1, q=1, rescale=False)
        res = am.fit(disp="off", show_warning=False)

    forecasts = res.forecast(horizon=horizon, reindex=False)
    # variance in last row, last column = h-step ahead
    h_step_var = forecasts.variance.iloc[-1, -1]

    # convert back to daily vol, then annualize
    daily_vol = np.sqrt(h_step_var) / scale
    return float(daily_vol * ANNUALIZATION_FACTOR)


def compute_iv_rv_spread(
    atm_iv: np.ndarray,
    rv_21d: np.ndarray,
) -> np.ndarray:
    """
    Implied-minus-realized volatility spread (vol points).

    Parameters
    ----------
    atm_iv : np.ndarray, shape (N,)
        At-the-money implied volatility (annualized fraction, e.g. 0.20).
    rv_21d : np.ndarray, shape (N,)
        21-day realized volatility (same units as atm_iv).

    Returns
    -------
    np.ndarray, shape (N,)
        IV - RV spread; positive means implied > realized (typical risk premium).
    """
    atm_iv = np.asarray(atm_iv, dtype=float)
    rv_21d = np.asarray(rv_21d, dtype=float)

    if atm_iv.shape != rv_21d.shape:
        raise ValueError("atm_iv and rv_21d must have the same shape")

    return atm_iv - rv_21d
