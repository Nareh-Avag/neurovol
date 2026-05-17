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
from scipy.stats import norm

logger = logging.getLogger(__name__)

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
# Batch computation
# ---------------------------------------------------------------------------

def compute_iv_batch(quotes_df: pd.DataFrame, rate: float) -> pd.DataFrame:
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
    """
    df = quotes_df.copy()

    required = {"underlying_price", "strike", "time_to_expiry", "option_type", "mid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"compute_iv_batch: missing columns {missing}")

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
