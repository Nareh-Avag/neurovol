"""
Implied volatility level features for the Neural Volatility Surface Forecasting System.

Functions cover: ATM IV extraction, IV rank/percentile, variance risk premium,
and full term-structure surface feature computation.
"""

import numpy as np
from scipy import stats
from typing import Optional


def compute_atm_iv(
    iv_surface: np.ndarray,
    strike_grid: np.ndarray,
    expiry_idx: int = 3,
) -> float:
    """
    Extract the at-the-money implied volatility at a given expiry index.

    ATM is defined as the strike closest to moneyness = 1.0 (i.e. K/S = 100).
    The strike_grid is assumed to contain moneyness values (K/S or log-moneyness).

    Parameters
    ----------
    iv_surface : np.ndarray, shape (n_expiries, n_strikes)
        Grid of implied volatilities.
    strike_grid : np.ndarray, shape (n_strikes,)
        Moneyness values corresponding to the strike axis.
    expiry_idx : int
        Index along the expiry axis to extract (default 3, typically ~30-day).

    Returns
    -------
    float
        ATM implied volatility at the requested expiry.
    """
    iv_surface = np.asarray(iv_surface, dtype=float)
    strike_grid = np.asarray(strike_grid, dtype=float)

    if expiry_idx >= iv_surface.shape[0]:
        raise IndexError(
            f"expiry_idx {expiry_idx} out of range for surface with "
            f"{iv_surface.shape[0]} expiries"
        )

    iv_slice = iv_surface[expiry_idx]           # shape (n_strikes,)

    # Moneyness closest to 1.0 (if absolute moneyness) or 0.0 (if log-moneyness)
    log_moneyness = np.log(strike_grid) if np.all(strike_grid > 0) else strike_grid
    atm_idx = int(np.argmin(np.abs(log_moneyness)))
    return float(iv_slice[atm_idx])


def compute_iv_rank(
    current_iv: float,
    historical_ivs: np.ndarray,
) -> float:
    """
    IV rank over trailing 252 days.

    IV Rank = (IV - IV_min) / (IV_max - IV_min)

    Parameters
    ----------
    current_iv : float
        Today's IV level.
    historical_ivs : np.ndarray, shape (N,)
        Historical IV values (typically the trailing 252-day window).

    Returns
    -------
    float
        IV rank in [0, 1]. Returns 0.5 if max == min (flat history).
    """
    historical_ivs = np.asarray(historical_ivs, dtype=float)
    iv_min = float(np.nanmin(historical_ivs))
    iv_max = float(np.nanmax(historical_ivs))

    if iv_max == iv_min:
        return 0.5

    return float(np.clip((current_iv - iv_min) / (iv_max - iv_min), 0.0, 1.0))


def compute_iv_percentile(
    current_iv: float,
    historical_ivs: np.ndarray,
) -> float:
    """
    IV percentile rank over the historical window.

    Returns the fraction of historical observations that are below current_iv.

    Parameters
    ----------
    current_iv : float
        Today's IV level.
    historical_ivs : np.ndarray, shape (N,)
        Historical IV values.

    Returns
    -------
    float
        Percentile in [0, 1].
    """
    historical_ivs = np.asarray(historical_ivs, dtype=float)
    valid = historical_ivs[~np.isnan(historical_ivs)]

    if len(valid) == 0:
        return 0.5

    return float(stats.percentileofscore(valid, current_iv, kind="mean") / 100.0)


def compute_vrp(
    atm_iv: float,
    rv_21d: float,
) -> float:
    """
    Variance risk premium (VRP) in variance units.

    VRP = IV^2 - RV^2

    Positive VRP indicates that the market prices volatility above realized,
    compensating variance sellers for bearing variance risk.

    Parameters
    ----------
    atm_iv : float
        ATM implied volatility (annualized, e.g. 0.20).
    rv_21d : float
        21-day realized volatility (same units).

    Returns
    -------
    float
        Variance risk premium.
    """
    return float(atm_iv ** 2 - rv_21d ** 2)


def compute_iv_term_surface_features(
    iv_surface: np.ndarray,
    strike_grid: np.ndarray,
    expiry_grid: np.ndarray,
) -> dict:
    """
    Compute a rich set of term-structure features from the full IV surface.

    Parameters
    ----------
    iv_surface : np.ndarray, shape (n_expiries, n_strikes)
        Implied volatility surface.
    strike_grid : np.ndarray, shape (n_strikes,)
        Moneyness values (K/S or log-moneyness).
    expiry_grid : np.ndarray, shape (n_expiries,)
        Expiry times in calendar days (e.g. [7, 14, 30, 60, 90, 180, 365]).

    Returns
    -------
    dict with keys:
        atm_iv_by_expiry : np.ndarray  — ATM IV at each expiry
        ts_slope         : float       — linear regression slope of ATM IV on log(T)
        ts_curvature     : float       — quadratic coefficient of ATM IV on log(T)
        ts_1m_3m         : float       — spread between 30d and 90d ATM IV
        ts_3m_6m         : float       — spread between 90d and 180d ATM IV
        ts_1m_6m         : float       — spread between 30d and 180d ATM IV
        fwd_vol_1m_3m    : float       — 1m×3m forward volatility
        is_inverted      : bool        — True if 30d IV > 90d IV
        contango_ratio   : float       — 90d IV / 30d IV
    """
    iv_surface = np.asarray(iv_surface, dtype=float)
    strike_grid = np.asarray(strike_grid, dtype=float)
    expiry_grid = np.asarray(expiry_grid, dtype=float)

    n_expiries, n_strikes = iv_surface.shape

    if len(expiry_grid) != n_expiries:
        raise ValueError("expiry_grid length must match iv_surface first dimension")
    if len(strike_grid) != n_strikes:
        raise ValueError("strike_grid length must match iv_surface second dimension")

    # Extract ATM IV at each expiry
    log_moneyness = np.log(strike_grid) if np.all(strike_grid > 0) else strike_grid
    atm_idx = int(np.argmin(np.abs(log_moneyness)))
    atm_iv_by_expiry = iv_surface[:, atm_idx].copy()  # shape (n_expiries,)

    log_expiry = np.log(np.maximum(expiry_grid, 1.0))  # avoid log(0)

    # Term structure slope (linear regression of ATM IV on log(T))
    ts_slope, ts_intercept, *_ = stats.linregress(log_expiry, atm_iv_by_expiry)

    # Term structure curvature (quadratic fit, second-degree coefficient)
    poly_coeffs = np.polyfit(log_expiry, atm_iv_by_expiry, deg=2)
    ts_curvature = float(poly_coeffs[0])  # coefficient of (log T)^2

    def _atm_at_expiry_days(target_days: float) -> Optional[float]:
        """Interpolate ATM IV at a target expiry in days."""
        idx = np.argmin(np.abs(expiry_grid - target_days))
        candidates = np.where(np.abs(expiry_grid - target_days) == np.abs(expiry_grid - target_days).min())[0]
        return float(atm_iv_by_expiry[candidates[0]])

    iv_1m = _atm_at_expiry_days(30.0)
    iv_3m = _atm_at_expiry_days(90.0)
    iv_6m = _atm_at_expiry_days(180.0)

    ts_1m_3m = float(iv_1m - iv_3m)
    ts_3m_6m = float(iv_3m - iv_6m)
    ts_1m_6m = float(iv_1m - iv_6m)

    # Forward vol: 1m into 3m
    T_front = 30.0 / 252.0
    T_back = 90.0 / 252.0
    var_back = iv_3m ** 2 * T_back
    var_front = iv_1m ** 2 * T_front
    fwd_var = (var_back - var_front) / max(T_back - T_front, 1e-8)
    fwd_vol_1m_3m = float(np.sqrt(max(fwd_var, 0.0)))

    is_inverted = bool(iv_1m > iv_3m)
    contango_ratio = float(iv_3m / iv_1m) if iv_1m > 0 else 1.0

    return {
        "atm_iv_by_expiry": atm_iv_by_expiry,
        "ts_slope": float(ts_slope),
        "ts_curvature": ts_curvature,
        "ts_1m_3m": ts_1m_3m,
        "ts_3m_6m": ts_3m_6m,
        "ts_1m_6m": ts_1m_6m,
        "fwd_vol_1m_3m": fwd_vol_1m_3m,
        "is_inverted": is_inverted,
        "contango_ratio": contango_ratio,
    }
