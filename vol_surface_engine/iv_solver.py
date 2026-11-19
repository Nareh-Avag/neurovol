"""
Implied Volatility Solver
=========================
Provides Black-Scholes pricing, Greeks, and multiple IV inversion methods
(Newton-Raphson and Brent's method) with a vectorised batch interface.
"""

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Optional C++ backend (built from cpp/ with CMake). If it is missing, the
# "cpp" backend falls back to the vectorized NumPy solver with a warning.
try:
    from . import _iv_core  # type: ignore[attr-defined]
    CPP_AVAILABLE = True
except ImportError:
    _iv_core = None
    CPP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Black-Scholes pricing
# ---------------------------------------------------------------------------

def black_scholes(S: float, K: float, T: float, r: float, sigma: float,
                  option_type: str) -> float:
    """
    Black-Scholes option price.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years (trading-day convention, e.g. days/252).
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Implied volatility (annual).
    option_type : str
        'C' for call, 'P' for put.

    Returns
    -------
    float
        Option price.  Returns intrinsic value when T <= 0 or sigma <= 0.
    """
    if T <= 0.0 or sigma <= 0.0:
        if option_type.upper() == "C":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type.upper() == "C":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# Greeks
# ---------------------------------------------------------------------------

def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vega (dPrice/dSigma)."""
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    return S * norm.pdf(d1) * sqrt_T


def bs_delta(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str) -> float:
    """Black-Scholes delta (dPrice/dS)."""
    if T <= 0.0 or sigma <= 0.0:
        if option_type.upper() == "C":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    if option_type.upper() == "C":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes gamma (d²Price/dS²)."""
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    return norm.pdf(d1) / (S * sigma * sqrt_T)


def bs_theta(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str) -> float:
    """
    Black-Scholes theta (dPrice/dTime) expressed per calendar day.
    Positive value means the option loses value as time passes.
    """
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    discount = math.exp(-r * T)

    term1 = -S * norm.pdf(d1) * sigma / (2.0 * sqrt_T)
    if option_type.upper() == "C":
        theta_annual = term1 - r * K * discount * norm.cdf(d2)
    else:
        theta_annual = term1 + r * K * discount * norm.cdf(-d2)
    return theta_annual / 365.0  # per calendar day


# ---------------------------------------------------------------------------
# Initial guess
# ---------------------------------------------------------------------------

def brenner_subrahmanyam_guess(price: float, S: float, K: float, T: float) -> float:
    """
    Brenner & Subrahmanyam (1988) ATM approximation as initial IV guess.

    sigma ≈ price / (S * sqrt(T / (2*pi)))

    Falls back to 0.2 when the formula is not applicable.
    """
    if S <= 0.0 or T <= 0.0:
        return 0.2
    guess = price / (S * math.sqrt(T / (2.0 * math.pi)))
    # Clamp to a sensible range
    return float(np.clip(guess, 0.01, 10.0))


# ---------------------------------------------------------------------------
# Newton-Raphson IV solver
# ---------------------------------------------------------------------------

def implied_vol_newton(price: float, S: float, K: float, T: float, r: float,
                       option_type: str, tol: float = 1e-8,
                       max_iter: int = 100) -> float:
    """
    Implied volatility via Newton-Raphson iteration.

    Returns NaN if the method fails to converge or if inputs are degenerate.
    """
    if price <= 0.0 or S <= 0.0 or K <= 0.0 or T <= 0.0:
        return float("nan")

    # Intrinsic value check
    if option_type.upper() == "C":
        intrinsic = max(S - K * math.exp(-r * T), 0.0)
    else:
        intrinsic = max(K * math.exp(-r * T) - S, 0.0)

    if price < intrinsic - 1e-6:
        logger.debug("Price %.4f below intrinsic %.4f; returning NaN", price, intrinsic)
        return float("nan")

    sigma = brenner_subrahmanyam_guess(price, S, K, T)

    for i in range(max_iter):
        price_est = black_scholes(S, K, T, r, sigma, option_type)
        vega = bs_vega(S, K, T, r, sigma)

        diff = price_est - price
        if abs(diff) < tol:
            return sigma

        if vega < 1e-10:
            logger.debug("Vega near zero at iter %d (sigma=%.4f); aborting Newton", i, sigma)
            return float("nan")

        sigma -= diff / vega
        sigma = float(np.clip(sigma, 1e-6, 20.0))

    logger.debug("Newton-Raphson did not converge after %d iterations", max_iter)
    return float("nan")


# ---------------------------------------------------------------------------
# Brent's method IV solver
# ---------------------------------------------------------------------------

def implied_vol_brent(price: float, S: float, K: float, T: float, r: float,
                      option_type: str, tol: float = 1e-8) -> float:
    """
    Implied volatility via Brent's method (scipy.optimize.brentq).

    Returns NaN if the method fails or inputs are degenerate.
    """
    if price <= 0.0 or S <= 0.0 or K <= 0.0 or T <= 0.0:
        return float("nan")

    def objective(sigma: float) -> float:
        return black_scholes(S, K, T, r, sigma, option_type) - price

    sigma_lo, sigma_hi = 1e-6, 20.0

    try:
        f_lo = objective(sigma_lo)
        f_hi = objective(sigma_hi)
        if f_lo * f_hi > 0:
            # Price not bracketed — possibly deep ITM/OTM
            logger.debug("Brent: objective does not change sign; f_lo=%.4f f_hi=%.4f",
                         f_lo, f_hi)
            return float("nan")
        iv = brentq(objective, sigma_lo, sigma_hi, xtol=tol, maxiter=200)
        return float(iv)
    except ValueError as exc:
        logger.debug("Brent method failed: %s", exc)
        return float("nan")


# ---------------------------------------------------------------------------
# Vectorized NumPy implementation
# ---------------------------------------------------------------------------
# Same Newton as implied_vol_newton above (same initial guess, clipping, and
# tolerance), but each iteration runs on a whole array at once and rows drop
# out of the active set as they converge. Rows where Newton gives up go to a
# vectorized bisection on the same [1e-6, 20] bracket that implied_vol_brent
# uses. Bisection is used instead of Brent because Brent's step choice is
# data dependent per row and does not vectorize cleanly. The result is within
# a few 1e-9 of the root, not bit-identical to scipy's brentq.

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _d1_vec(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def black_scholes_vec(S, K, T, r, sigma, is_call):
    """Vectorized Black-Scholes price. Assumes T > 0 and sigma > 0."""
    S, K, T, sigma = (np.asarray(a, dtype=np.float64) for a in (S, K, T, sigma))
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    disc_K = K * np.exp(-r * T)
    call = S * ndtr(d1) - disc_K * ndtr(d2)
    put = disc_K * ndtr(-d2) - S * ndtr(-d1)
    return np.where(is_call, call, put)


def bs_vega_vec(S, K, T, r, sigma):
    """Vectorized Black-Scholes vega. Assumes T > 0 and sigma > 0."""
    d1 = _d1_vec(S, K, T, r, sigma)
    return S * np.exp(-0.5 * d1 * d1) / _SQRT_2PI * np.sqrt(T)


def implied_vol_vectorized(price, S, K, T, r: float, is_call,
                           tol: float = 1e-8, max_iter: int = 100) -> np.ndarray:
    """
    Vectorized Newton-Raphson IV with a vectorized bisection fallback.

    Parameters are 1-D arrays of equal length (``r`` is a scalar). ``is_call``
    is a boolean array. Returns a float64 array with NaN where no IV exists.
    """
    price, S, K, T = (np.asarray(a, dtype=np.float64) for a in (price, S, K, T))
    is_call = np.asarray(is_call, dtype=bool)
    out = np.full(price.shape, np.nan)

    with np.errstate(all="ignore"):
        valid = (np.isfinite(price) & np.isfinite(S) & np.isfinite(K) & np.isfinite(T)
                 & (price > 0) & (S > 0) & (K > 0) & (T > 0))
        disc_K = K * np.exp(-r * T)
        intrinsic = np.where(is_call, np.maximum(S - disc_K, 0.0),
                             np.maximum(disc_K - S, 0.0))
        valid &= ~(price < intrinsic - 1e-6)

        idx = np.flatnonzero(valid)
        sigma = np.clip(price[idx] / (S[idx] * np.sqrt(T[idx] / (2.0 * math.pi))), 0.01, 10.0)
        failed = []

        for _ in range(max_iter):
            if idx.size == 0:
                break
            s, k, t, c = S[idx], K[idx], T[idx], is_call[idx]
            diff = black_scholes_vec(s, k, t, r, sigma, c) - price[idx]
            vega = bs_vega_vec(s, k, t, r, sigma)

            conv = np.abs(diff) < tol
            out[idx[conv]] = sigma[conv]
            dead = ~conv & (vega < 1e-10)
            failed.append(idx[dead])

            keep = ~conv & ~dead
            idx, sigma, diff, vega = idx[keep], sigma[keep], diff[keep], vega[keep]
            sigma = np.clip(sigma - diff / vega, 1e-6, 20.0)
        failed.append(idx)  # still active after max_iter: did not converge

        # Fallback, also vectorized: bisection on the same bracket that
        # implied_vol_brent uses. Price is increasing in sigma, so the sign
        # of BS(mid) - price says which half holds the root. 32 halvings of
        # [1e-6, 20] leave a bracket of 20 / 2**32 = 4.7e-9, and the midpoint
        # is within 2.3e-9 of the root.
        fb = np.concatenate(failed)
        if fb.size:
            s, k, t, c, p = S[fb], K[fb], T[fb], is_call[fb], price[fb]
            lo = np.full(fb.size, 1e-6)
            hi = np.full(fb.size, 20.0)
            f_lo = black_scholes_vec(s, k, t, r, lo, c) - p
            f_hi = black_scholes_vec(s, k, t, r, hi, c) - p
            bracketed = f_lo * f_hi <= 0.0
            for _ in range(32):
                mid = 0.5 * (lo + hi)
                above = black_scholes_vec(s, k, t, r, mid, c) - p > 0.0
                hi = np.where(above, mid, hi)
                lo = np.where(above, lo, mid)
            out[fb] = np.where(bracketed, 0.5 * (lo + hi), np.nan)
    return out


def _implied_vol_rows_python(price, S, K, T, r, is_call) -> np.ndarray:
    """Original per-row path: Newton, then Brent if Newton returns NaN."""
    out = np.empty(len(price))
    for i in range(len(price)):
        opt = "C" if is_call[i] else "P"
        p, s, k, t = float(price[i]), float(S[i]), float(K[i]), float(T[i])
        iv = implied_vol_newton(p, s, k, t, r, opt)
        if math.isnan(iv):
            iv = implied_vol_brent(p, s, k, t, r, opt)
        out[i] = iv
    return out


def implied_vol_array(price, S, K, T, r: float, is_call,
                      backend: str = "numpy") -> np.ndarray:
    """
    Solve IV for arrays of quotes with the chosen backend.

    backend
        "python": original per-row scalar solver (slow, reference).
        "numpy":  vectorized NumPy solver.
        "cpp":    C++ extension; falls back to "numpy" with a warning when the
                  extension has not been built.
    """
    if backend == "python":
        return _implied_vol_rows_python(price, S, K, T, r, is_call)
    if backend == "numpy":
        return implied_vol_vectorized(price, S, K, T, r, is_call)
    if backend == "cpp":
        if not CPP_AVAILABLE:
            logger.warning("C++ backend not built; falling back to NumPy backend")
            return implied_vol_vectorized(price, S, K, T, r, is_call)
        # np.ascontiguousarray is a no-op (no copy) when the input is already
        # a C-contiguous float64 / uint8 array.
        price, S, K, T = (np.ascontiguousarray(a, dtype=np.float64) for a in (price, S, K, T))
        flags = np.ascontiguousarray(is_call, dtype=np.uint8)
        out = np.empty(price.shape[0], dtype=np.float64)
        _iv_core.implied_vol_batch(S, K, T, price, flags, float(r), out)
        return out
    raise ValueError(f"unknown backend {backend!r}")


def _greeks_vec(S, K, T, r, sigma, is_call):
    """Vectorized delta, gamma, vega, theta (per calendar day)."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    pdf = np.exp(-0.5 * d1 * d1) / _SQRT_2PI
    disc = np.exp(-r * T)
    delta = np.where(is_call, ndtr(d1), ndtr(d1) - 1.0)
    gamma = pdf / (S * sigma * sqrt_T)
    vega = S * pdf * sqrt_T
    term1 = -S * pdf * sigma / (2.0 * sqrt_T)
    theta = np.where(is_call, term1 - r * K * disc * ndtr(d2),
                     term1 + r * K * disc * ndtr(-d2)) / 365.0
    return delta, gamma, vega, theta


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------

def compute_iv_batch(quotes_df: pd.DataFrame, rate: float,
                     backend: str = "python") -> pd.DataFrame:
    """
    Compute implied volatility and Greeks for every row in *quotes_df*.

    Expected columns
    ----------------
    underlying_price, strike, time_to_expiry, option_type, mid (option price).

    Added columns (in place on a copy)
    ------------------------------------
    implied_vol, delta, gamma, vega, theta, log_moneyness, iv_valid

    The function attempts Newton-Raphson first; if that returns NaN it falls
    back to Brent's method.  Rows that still yield NaN after both attempts
    get ``iv_valid = False``.

    backend
        "python" (default): original per-row implementation.
        "numpy": vectorized NumPy IV and Greeks.
        "cpp": C++ IV solver plus vectorized NumPy Greeks. Falls back to
        "numpy" if the C++ extension is not built.
    """
    required = {"underlying_price", "strike", "time_to_expiry", "option_type", "mid"}
    missing = required - set(quotes_df.columns)
    if missing:
        raise ValueError(f"compute_iv_batch: missing columns {missing}")

    if backend != "python":
        return _compute_iv_batch_vectorized(quotes_df, rate, backend)

    df = quotes_df.copy()

    ivs, deltas, gammas, vegas, thetas, log_ms, valid_flags = [], [], [], [], [], [], []

    for _, row in df.iterrows():
        S = float(row["underlying_price"])
        K = float(row["strike"])
        T = float(row["time_to_expiry"])
        r = rate
        opt = str(row["option_type"]).upper()
        price = float(row["mid"])

        # Log-moneyness
        if S > 0 and K > 0:
            lm = math.log(K / S)
        else:
            lm = 0.0

        # IV — try Newton then Brent
        iv = implied_vol_newton(price, S, K, T, r, opt)
        if math.isnan(iv):
            iv = implied_vol_brent(price, S, K, T, r, opt)

        if not math.isnan(iv) and iv > 0:
            d = bs_delta(S, K, T, r, iv, opt)
            g = bs_gamma(S, K, T, r, iv)
            v = bs_vega(S, K, T, r, iv)
            th = bs_theta(S, K, T, r, iv, opt)
            is_valid = True
        else:
            iv, d, g, v, th = float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
            is_valid = False

        ivs.append(iv)
        deltas.append(d)
        gammas.append(g)
        vegas.append(v)
        thetas.append(th)
        log_ms.append(lm)
        valid_flags.append(is_valid)

    df["implied_vol"] = ivs
    df["delta"] = deltas
    df["gamma"] = gammas
    df["vega"] = vegas
    df["theta"] = thetas
    df["log_moneyness"] = log_ms
    df["iv_valid"] = valid_flags

    n_valid = sum(valid_flags)
    logger.info("compute_iv_batch: %d/%d rows with valid IV", n_valid, len(df))
    return df


def _compute_iv_batch_vectorized(quotes_df: pd.DataFrame, rate: float,
                                 backend: str) -> pd.DataFrame:
    """compute_iv_batch for the "numpy" and "cpp" backends."""
    df = quotes_df.copy()
    S = df["underlying_price"].to_numpy(dtype=np.float64)
    K = df["strike"].to_numpy(dtype=np.float64)
    T = df["time_to_expiry"].to_numpy(dtype=np.float64)
    price = df["mid"].to_numpy(dtype=np.float64)
    is_call = df["option_type"].astype(str).str.upper().to_numpy() == "C"

    iv = implied_vol_array(price, S, K, T, rate, is_call, backend=backend)
    valid = np.isfinite(iv) & (iv > 0)

    nan = np.full(len(df), np.nan)
    delta, gamma, vega, theta = nan.copy(), nan.copy(), nan.copy(), nan.copy()
    if valid.any():
        v = valid
        delta[v], gamma[v], vega[v], theta[v] = _greeks_vec(
            S[v], K[v], T[v], rate, iv[v], is_call[v])

    with np.errstate(divide="ignore", invalid="ignore"):
        log_m = np.where((S > 0) & (K > 0), np.log(K / S), 0.0)

    df["implied_vol"] = np.where(valid, iv, np.nan)
    df["delta"] = delta
    df["gamma"] = gamma
    df["vega"] = vega
    df["theta"] = theta
    df["log_moneyness"] = log_m
    df["iv_valid"] = valid
    logger.info("compute_iv_batch[%s]: %d/%d rows with valid IV",
                backend, int(valid.sum()), len(df))
    return df
