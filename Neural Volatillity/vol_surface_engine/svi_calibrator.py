"""
SVI Calibrator
==============
Fits the Stochastic Volatility Inspired (SVI) parametrisation to market
implied volatility slices.

SVI total variance:
    w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + sigma^2) )

where k = log(K/F) is log-moneyness.

Constraints (no-arbitrage necessary conditions):
    b >= 0
    |rho| < 1
    sigma > 0
    a + b*sigma*sqrt(1 - rho^2) >= 0   (ATM density non-negative)
"""

import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from .schemas import SVIParams

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SVI formula helpers
# ---------------------------------------------------------------------------

def svi_total_variance(k: float, params: SVIParams) -> float:
    """Evaluate SVI total variance w(k) at a single log-moneyness point."""
    return params.a + params.b * (
        params.rho * (k - params.m)
        + np.sqrt((k - params.m) ** 2 + params.sigma ** 2)
    )


def _svi_vec(k: np.ndarray, a: float, b: float, rho: float,
             m: float, sigma: float) -> np.ndarray:
    """Vectorised SVI total variance (internal use)."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


# ---------------------------------------------------------------------------
# Butterfly arbitrage check
# ---------------------------------------------------------------------------

def check_butterfly_arbitrage(params: SVIParams, k_grid: np.ndarray) -> bool:
    """
    Check that the SVI slice is free of butterfly arbitrage.

    Butterfly arbitrage is absent iff the implied risk-neutral density
    g(k) >= 0 for all k.  We approximate g(k) via the Gatheral-Jacquier
    formula:

        g(k) = (1 - k*w'/(2w))^2 - (w'/4)*(1/w + 1/4) + w''/2

    where w = w(k), w' = dw/dk, w'' = d²w/dk².

    Returns True if no butterfly arbitrage is detected on k_grid.
    """
    k = k_grid.astype(float)
    a, b, rho, m, sigma = params.a, params.b, params.rho, params.m, params.sigma

    # First and second derivatives of SVI w.r.t. k (analytically)
    diff = k - m
    sq = np.sqrt(diff ** 2 + sigma ** 2)

    w = _svi_vec(k, a, b, rho, m, sigma)
    w_prime = b * (rho + diff / sq)
    w_double = b * sigma ** 2 / sq ** 3

    # Gatheral density (unnormalised)
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = (1.0 - k * w_prime / (2.0 * w)) ** 2
        term2 = w_prime ** 2 / 4.0 * (1.0 / w + 0.25)
        g = term1 - term2 + w_double / 2.0

    min_g = float(np.nanmin(g))
    if min_g < -1e-6:
        logger.debug(
            "Butterfly arbitrage detected: min g(k) = %.6f at %d nodes", min_g, len(k)
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Slice calibration
# ---------------------------------------------------------------------------

def _objective(params_vec: np.ndarray, k: np.ndarray, w_market: np.ndarray,
               weights: np.ndarray) -> float:
    a, b, rho, m, sigma = params_vec
    w_model = _svi_vec(k, a, b, rho, m, sigma)
    residuals = weights * (w_model - w_market) ** 2
    return float(np.sum(residuals))


class SVICalibrator:
    """Calibrates SVI parameters to market implied-volatility slices."""

    # Default parameter bounds:  (a, b, rho, m, sigma)
    _BOUNDS = [
        (-2.0, 2.0),    # a   – intercept
        (1e-6, 5.0),    # b   – slope must be non-negative
        (-0.999, 0.999),# rho – correlation
        (-2.0, 2.0),    # m   – ATM level in log-moneyness
        (1e-6, 5.0),    # sigma – curvature must be positive
    ]

    def fit_slice(
        self,
        log_moneyness: np.ndarray,
        total_variance: np.ndarray,
        weights: Optional[np.ndarray] = None,
        expiry_days: int = 30,
    ) -> SVIParams:
        """
        Fit SVI to a single expiry slice.

        Parameters
        ----------
        log_moneyness : np.ndarray
            k = log(K/F) values.
        total_variance : np.ndarray
            w = sigma_iv^2 * T values.
        weights : np.ndarray, optional
            Per-point weights.  Defaults to uniform.
        expiry_days : int
            Calendar days to expiry (stored in SVIParams for reference).

        Returns
        -------
        SVIParams
            Best-fit SVI parameters.
        """
        k = np.asarray(log_moneyness, dtype=float)
        w = np.asarray(total_variance, dtype=float)

        if weights is None:
            wt = np.ones_like(k)
        else:
            wt = np.asarray(weights, dtype=float)

        # Clip total variance to avoid fitting noise
        w = np.clip(w, 1e-8, None)

        # Initial guess: flat vol surface
        atm_w = np.median(w)
        x0 = np.array([atm_w * 0.9, 0.1, -0.1, 0.0, 0.1])

        best_result = None
        best_cost = np.inf

        # Multiple random restarts for robustness
        rng = np.random.default_rng(42)
        starts = [x0] + [
            np.array([
                rng.uniform(-0.5, 0.5),
                rng.uniform(0.01, 0.5),
                rng.uniform(-0.8, 0.8),
                rng.uniform(-0.5, 0.5),
                rng.uniform(0.01, 0.5),
            ])
            for _ in range(4)
        ]

        for start in starts:
            result = minimize(
                _objective,
                start,
                args=(k, w, wt),
                method="L-BFGS-B",
                bounds=self._BOUNDS,
                options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10},
            )
            if result.success or result.fun < best_cost:
                if result.fun < best_cost:
                    best_cost = result.fun
                    best_result = result

        if best_result is None:
            logger.warning("SVI fit failed for expiry_days=%d; using flat surface", expiry_days)
            return SVIParams(a=atm_w, b=0.0, rho=0.0, m=0.0, sigma=0.1,
                             expiry_days=expiry_days)

        a, b, rho, m, sigma = best_result.x
        params = SVIParams(a=float(a), b=float(b), rho=float(rho),
                           m=float(m), sigma=float(sigma),
                           expiry_days=expiry_days)

        logger.info(
            "SVI fit expiry_days=%d  cost=%.2e  a=%.4f b=%.4f rho=%.4f m=%.4f sigma=%.4f",
            expiry_days, best_cost, a, b, rho, m, sigma,
        )
        return params

    # ------------------------------------------------------------------
    # Surface calibration
    # ------------------------------------------------------------------

    def calibrate_surface(
        self,
        iv_quotes: list,
        expiry_days_list: list,
    ) -> dict:
        """
        Calibrate one SVI slice per expiry.

        Parameters
        ----------
        iv_quotes : list of dict
            Each element must contain keys:
              ``log_moneyness``, ``total_variance``, ``expiry_days``,
              and optionally ``weight``.
        expiry_days_list : list of int
            Expiry buckets to calibrate.

        Returns
        -------
        dict
            ``{expiry_days: SVIParams}``
        """
        # Group quotes by expiry
        groups: dict[int, list] = {d: [] for d in expiry_days_list}
        for q in iv_quotes:
            ed = int(q["expiry_days"])
            # Snap to nearest requested expiry
            nearest = min(expiry_days_list, key=lambda x: abs(x - ed))
            groups[nearest].append(q)

        svi_dict: dict[int, SVIParams] = {}

        for exp_days in expiry_days_list:
            slice_quotes = groups[exp_days]
            if len(slice_quotes) < 3:
                logger.warning(
                    "Insufficient quotes (%d) for expiry_days=%d; skipping",
                    len(slice_quotes), exp_days,
                )
                continue

            k_arr = np.array([q["log_moneyness"] for q in slice_quotes])
            tv_arr = np.array([q["total_variance"] for q in slice_quotes])
            wt_arr = np.array([q.get("weight", 1.0) for q in slice_quotes])

            # Filter out NaN / inf
            mask = np.isfinite(k_arr) & np.isfinite(tv_arr) & (tv_arr > 0)
            if mask.sum() < 3:
                logger.warning(
                    "Too few valid quotes after filtering for expiry_days=%d", exp_days
                )
                continue

            params = self.fit_slice(k_arr[mask], tv_arr[mask], wt_arr[mask], exp_days)

            # Butterfly check
            k_grid = np.linspace(k_arr[mask].min() - 0.05, k_arr[mask].max() + 0.05, 200)
            arb_free = check_butterfly_arbitrage(params, k_grid)
            if not arb_free:
                logger.warning(
                    "Butterfly arbitrage in SVI fit for expiry_days=%d", exp_days
                )

            svi_dict[exp_days] = params

        logger.info("SVI calibration complete: %d/%d slices fitted",
                    len(svi_dict), len(expiry_days_list))
        return svi_dict
