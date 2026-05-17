"""
P&L attribution for the Neural Volatility Surface Forecasting System.

Breaks down daily option portfolio P&L into first- and second-order Greek
contributions, computes a residual, and aggregates rolling risk metrics.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PnLAttributor:
    """
    Attributes realised P&L to individual option Greeks and computes
    portfolio-level risk statistics.

    The Taylor-expansion used is:

        ΔV ≈  Δ·ΔS
              + ½·Γ·ΔS²
              + ν·Δσ
              + Θ·Δt
              + vanna·ΔS·Δσ
              + ½·volga·Δσ²

    where Δt is measured in years (default 1/252 for one business day).
    """

    # ------------------------------------------------------------------
    # Greek P&L decomposition
    # ------------------------------------------------------------------

    @staticmethod
    def greek_pnl_breakdown(
        net_delta: float,
        net_gamma: float,
        net_vega: float,
        net_theta: float,
        net_vanna: float,
        net_volga: float,
        dS: float,
        d_sigma: float,
        dt: float = 1 / 252,
    ) -> dict[str, float]:
        """
        Compute the Greek P&L breakdown for a single period.

        Parameters
        ----------
        net_delta : float
            Portfolio delta (∂V/∂S).
        net_gamma : float
            Portfolio gamma (∂²V/∂S²).
        net_vega : float
            Portfolio vega (∂V/∂σ), in vol-point units.
        net_theta : float
            Portfolio theta (∂V/∂t), negative for long optionality.
        net_vanna : float
            Portfolio vanna (∂²V/∂S∂σ).
        net_volga : float
            Portfolio volga / vomma (∂²V/∂σ²).
        dS : float
            Change in underlying price over the period.
        d_sigma : float
            Change in implied volatility over the period (in vol-point units,
            e.g. 0.01 = 1 vol point).
        dt : float
            Time elapsed in years.  Default is one business day (1/252).

        Returns
        -------
        dict[str, float]
            Keys: delta_pnl, gamma_pnl, vega_pnl, theta_pnl,
                  vanna_pnl, volga_pnl.
        """
        delta_pnl: float = net_delta * dS
        gamma_pnl: float = 0.5 * net_gamma * dS ** 2
        vega_pnl: float = net_vega * d_sigma
        theta_pnl: float = net_theta * dt
        vanna_pnl: float = net_vanna * dS * d_sigma
        volga_pnl: float = 0.5 * net_volga * d_sigma ** 2

        breakdown = {
            "delta_pnl": delta_pnl,
            "gamma_pnl": gamma_pnl,
            "vega_pnl": vega_pnl,
            "theta_pnl": theta_pnl,
            "vanna_pnl": vanna_pnl,
            "volga_pnl": volga_pnl,
        }
        logger.debug("Greek P&L breakdown: %s", breakdown)
        return breakdown

    # ------------------------------------------------------------------
    # Daily summary
    # ------------------------------------------------------------------

    @staticmethod
    def daily_summary(actual_pnl: float, greek_breakdown: dict) -> dict[str, float]:
        """
        Enrich the Greek breakdown with total Greek P&L, residual, and actual P&L.

        Parameters
        ----------
        actual_pnl : float
            Realised P&L for the day (e.g. from mark-to-market).
        greek_breakdown : dict
            Output of :meth:`greek_pnl_breakdown`.

        Returns
        -------
        dict[str, float]
            greek_breakdown fields + {total_greek_pnl, residual_pnl, actual_pnl}.
        """
        total_greek_pnl: float = sum(greek_breakdown.values())
        residual_pnl: float = actual_pnl - total_greek_pnl

        summary = {
            **greek_breakdown,
            "total_greek_pnl": total_greek_pnl,
            "residual_pnl": residual_pnl,
            "actual_pnl": actual_pnl,
        }
        logger.debug(
            "Daily summary: actual=%.2f total_greek=%.2f residual=%.2f",
            actual_pnl, total_greek_pnl, residual_pnl,
        )
        return summary

    # ------------------------------------------------------------------
    # Rolling risk metrics
    # ------------------------------------------------------------------

    @staticmethod
    def rolling_sharpe(
        daily_pnls: np.ndarray,
        window: int = 63,
        annualisation: int = 252,
    ) -> np.ndarray:
        """
        Compute a rolling Sharpe ratio.

        The first ``window - 1`` values are set to NaN (insufficient history).

        Parameters
        ----------
        daily_pnls : np.ndarray
            1-D array of daily P&L values.
        window : int
            Look-back window in business days.  Default 63 (≈ 1 quarter).
        annualisation : int
            Number of business days in a year.  Default 252.

        Returns
        -------
        np.ndarray
            Rolling Sharpe ratio, same length as input.
        """
        pnls = np.asarray(daily_pnls, dtype=float)
        n = len(pnls)
        sharpe = np.full(n, np.nan)

        for i in range(window - 1, n):
            window_pnl = pnls[i - window + 1 : i + 1]
            mu = np.mean(window_pnl)
            sigma = np.std(window_pnl, ddof=1)
            if sigma > 1e-12:
                sharpe[i] = (mu / sigma) * np.sqrt(annualisation)
            else:
                sharpe[i] = 0.0

        return sharpe

    # ------------------------------------------------------------------
    # Drawdown metrics
    # ------------------------------------------------------------------

    @staticmethod
    def max_drawdown(cumulative_pnl: np.ndarray) -> float:
        """
        Compute the maximum drawdown of a cumulative P&L series.

        Parameters
        ----------
        cumulative_pnl : np.ndarray
            Running total P&L (not returns).

        Returns
        -------
        float
            Maximum peak-to-trough drawdown (expressed as a positive number).
        """
        cum = np.asarray(cumulative_pnl, dtype=float)
        running_max = np.maximum.accumulate(cum)
        drawdowns = running_max - cum
        mdd = float(np.max(drawdowns))
        logger.debug("Max drawdown: %.4f", mdd)
        return mdd

    # ------------------------------------------------------------------
    # Return ratios
    # ------------------------------------------------------------------

    @staticmethod
    def calmar_ratio(annual_return: float, max_drawdown: float) -> float:
        """
        Compute the Calmar ratio.

        Parameters
        ----------
        annual_return : float
            Annualised return (absolute dollar or percentage – must be
            consistent with max_drawdown units).
        max_drawdown : float
            Maximum drawdown (positive number, same units as annual_return).

        Returns
        -------
        float
            Calmar ratio.  Returns 0.0 if max_drawdown is zero.
        """
        if max_drawdown < 1e-12:
            logger.warning("max_drawdown ≈ 0; returning Calmar ratio of 0.")
            return 0.0
        calmar = annual_return / max_drawdown
        logger.debug("Calmar ratio: %.4f", calmar)
        return calmar
