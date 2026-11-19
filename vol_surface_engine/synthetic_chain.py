"""
Deterministic synthetic options chain generator.

Used by the tests and by bench/benchmark.py. Every contract is priced with
Black-Scholes at a *known* volatility, so solvers can be checked against
ground truth.

Layout
------
A chain of ``n_contracts`` is built from ``n_contracts // 1000`` independent
underlyings (minimum 1). Each underlying gets:

* a spot price drawn from U(20, 500)
* an ATM volatility drawn from U(0.12, 0.60)
* a negative skew and a positive curvature (an equity-style smile)
* 20 expiries from 1 trading day to 2 years
* 25 strikes per expiry, evenly spaced in log-moneyness across
  +/- 2.5 standard deviations (atm_vol * sqrt(T)), so short expiries are
  tight and long expiries are wide, like a real listed chain
* a call and a put at every strike

20 expiries x 25 strikes x 2 types = 1000 contracts per underlying.

The smile is ``sigma(x) = atm + skew * x + curv * x**2`` with
``x = log(K / F) / sqrt(T)`` (so the smile flattens in log-strike as
maturity grows, as equity smiles do) and a floor of 0.05.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .iv_solver import black_scholes_vec

CONTRACTS_PER_UNDERLYING = 1000
_EXPIRY_DAYS = np.array(
    [1, 2, 5, 10, 15, 21, 30, 42, 63, 84, 105, 126, 147, 168, 189, 210, 252, 315, 378, 504],
    dtype=float,
)
_N_STRIKES = 25


def generate_chain(n_contracts: int, rate: float = 0.03, seed: int = 12345) -> pd.DataFrame:
    """
    Build a deterministic synthetic chain.

    Parameters
    ----------
    n_contracts : int
        Must be a positive multiple of 1000.
    rate : float
        Continuously compounded risk-free rate used for pricing.
    seed : int
        Seed for numpy's PCG64 generator. Same seed gives identical output.

    Returns
    -------
    pd.DataFrame
        Columns: underlying_id, underlying_price, strike, time_to_expiry,
        option_type ('C' or 'P'), mid, true_iv.
    """
    if n_contracts <= 0 or n_contracts % CONTRACTS_PER_UNDERLYING != 0:
        raise ValueError("n_contracts must be a positive multiple of 1000")

    rng = np.random.default_rng(seed)
    n_und = n_contracts // CONTRACTS_PER_UNDERLYING

    spots = rng.uniform(20.0, 500.0, n_und)
    atm_vols = rng.uniform(0.12, 0.60, n_und)
    skews = rng.uniform(-0.15, -0.03, n_und)
    curvs = rng.uniform(0.02, 0.15, n_und)

    T_years = _EXPIRY_DAYS / 252.0
    z = np.linspace(-2.5, 2.5, _N_STRIKES)  # strike placement in standard deviations

    # Broadcast to (underlying, expiry, strike)
    S = spots[:, None, None]
    T = T_years[None, :, None]
    atm = atm_vols[:, None, None]
    F = S * np.exp(rate * T)
    k = z[None, None, :] * atm * np.sqrt(T)              # log(K/F)
    K = F * np.exp(k)
    x = k / np.sqrt(T)                                    # time-scaled moneyness
    sigma = np.maximum(atm + skews[:, None, None] * x + curvs[:, None, None] * x**2, 0.05)

    shape = (n_und, len(T_years), _N_STRIKES)
    S_b, T_b, K_b, sig_b = (np.broadcast_to(a, shape).ravel() for a in (S, T, K, sigma))
    und_id = np.broadcast_to(np.arange(n_und)[:, None, None], shape).ravel()

    # Interleave calls and puts: every strike appears twice.
    S_all = np.repeat(S_b, 2)
    K_all = np.repeat(K_b, 2)
    T_all = np.repeat(T_b, 2)
    sig_all = np.repeat(sig_b, 2)
    is_call = np.tile(np.array([True, False]), S_b.size)

    price = black_scholes_vec(S_all, K_all, T_all, rate, sig_all, is_call)

    return pd.DataFrame(
        {
            "underlying_id": np.repeat(und_id, 2),
            "underlying_price": S_all,
            "strike": K_all,
            "time_to_expiry": T_all,
            "option_type": np.where(is_call, "C", "P"),
            "mid": price,
            "true_iv": sig_all,
        }
    )
