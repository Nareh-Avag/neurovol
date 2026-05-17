"""
Backtesting engine for the Neural Volatility Surface Forecasting System.

Iterates over a historical sequence of surfaces and forecasts, generates
trading signals, simulates simplified vol-carry trades, tracks P&L and
computes portfolio-level performance statistics.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    from rich.console import Console
    from rich.table import Table

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False

from backtesting.signal_generator import SignalGenerator
from backtesting.pnl_attribution import PnLAttributor

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Event-driven backtest engine for volatility-surface trading signals.

    Parameters
    ----------
    config : dict
        Accepted keys:
          - signal_config (dict): forwarded to SignalGenerator.
          - vega_notional (float): default vega notional per trade.
            Default 10 000.
          - annualisation (int): business days per year.  Default 252.
          - min_signal_threshold (float): ignore signals below this absolute
            value.  Default 0.10.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        signal_config: dict = config.get("signal_config", {})
        self.signal_generator = SignalGenerator(signal_config)
        self.pnl_attributor = PnLAttributor()
        self.vega_notional: float = float(config.get("vega_notional", 10_000))
        self.annualisation: int = int(config.get("annualisation", 252))
        self.min_signal_threshold: float = float(
            config.get("min_signal_threshold", 0.10)
        )
        logger.info("BacktestEngine initialised with config: %s", config)

    # ------------------------------------------------------------------
    # Main event loop
    # ------------------------------------------------------------------

    def run(
        self,
        surface_history: list[dict],
        forecast_history: list[dict],
        dates: list,
    ) -> pd.DataFrame:
        """
        Run the backtest over all dates.

        Parameters
        ----------
        surface_history : list[dict]
            One dict per date containing current surface statistics.
            Required keys: atm_iv, rv_21d, skew, term_slope.
        forecast_history : list[dict]
            One dict per date containing model forecast statistics.
            Required keys: atm_iv, skew, term_slope, inversion_prob.
        dates : list
            Sequence of date labels (str or datetime) aligned with the above.

        Returns
        -------
        pd.DataFrame
            Daily results with columns: date, signal, daily_pnl, cumulative_pnl.
        """
        if not (len(surface_history) == len(forecast_history) == len(dates)):
            raise ValueError(
                "surface_history, forecast_history and dates must have the same length."
            )

        records: list[dict[str, Any]] = []

        for date, surface_stats, forecast in zip(dates, surface_history, forecast_history):
            signals = self.signal_generator.generate_all(forecast, surface_stats)
            composite_signal: float = signals["composite"]

            atm_iv: float = float(surface_stats.get("atm_iv", 0.20))
            # Next-day realised vol approximated by the forecast ATM IV for
            # simulation purposes (in a real back-test this would be the
            # next observation).
            rv_realised: float = float(forecast.get("rv_realized", atm_iv * 0.92))

            daily_pnl: float = 0.0
            if abs(composite_signal) >= self.min_signal_threshold:
                daily_pnl = self.simulate_vol_carry_trade(
                    signal=composite_signal,
                    atm_iv=atm_iv,
                    rv_realized=rv_realised,
                    vega_notional=self.vega_notional,
                )

            records.append(
                {
                    "date": date,
                    "vol_carry_signal": signals["vol_carry"],
                    "skew_signal": signals["skew"],
                    "term_structure_signal": signals["term_structure"],
                    "inversion_signal": signals["inversion"],
                    "composite_signal": composite_signal,
                    "atm_iv": atm_iv,
                    "daily_pnl": daily_pnl,
                }
            )
            logger.debug("date=%s composite_signal=%.4f daily_pnl=%.2f", date, composite_signal, daily_pnl)

        results = pd.DataFrame(records)
        if not results.empty:
            results["cumulative_pnl"] = results["daily_pnl"].cumsum()

        logger.info("Backtest complete. Total P&L: %.2f", results["cumulative_pnl"].iloc[-1] if not results.empty else 0)
        return results

    # ------------------------------------------------------------------
    # Trade simulation
    # ------------------------------------------------------------------

    @staticmethod
    def simulate_vol_carry_trade(
        signal: float,
        atm_iv: float,
        rv_realized: float,
        vega_notional: float = 10_000,
    ) -> float:
        """
        Simplified vol-carry trade P&L simulation.

        P&L ≈ (rv_realized − atm_iv) × vega_notional × signal_direction

        A positive *signal* means we are short vol (sold at atm_iv and
        will profit if realised vol is lower).  The sign convention follows:

            pnl = (rv_realized - atm_iv) × vega_notional × (−sign(signal))

        because selling vol at atm_iv profits when rv_realized < atm_iv.

        Parameters
        ----------
        signal : float
            Composite signal in [-1, 1].  Positive → short vol.
        atm_iv : float
            ATM implied vol at trade initiation.
        rv_realized : float
            Subsequent realised vol over the holding period.
        vega_notional : float
            Dollar vega notional.  Default 10 000.

        Returns
        -------
        float
            Simulated trade P&L in dollars.
        """
        signal_direction: float = np.sign(signal)
        # Short vol profits when RV < IV: pnl = (atm_iv - rv_realized) for shorts
        vol_diff: float = atm_iv - rv_realized
        pnl: float = vol_diff * vega_notional * signal_direction * abs(signal)
        return float(pnl)

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    def compute_performance_metrics(self, daily_pnls: pd.Series) -> dict[str, float]:
        """
        Compute standard performance metrics from a daily P&L series.

        Parameters
        ----------
        daily_pnls : pd.Series
            Daily P&L values.

        Returns
        -------
        dict[str, float]
            Keys: sharpe, sortino, max_drawdown, hit_rate, calmar,
                  annual_return, total_pnl.
        """
        pnls = daily_pnls.dropna().to_numpy(dtype=float)

        if len(pnls) == 0:
            logger.warning("Empty P&L series — returning zero metrics.")
            return {k: 0.0 for k in ("sharpe", "sortino", "max_drawdown", "hit_rate", "calmar", "annual_return", "total_pnl")}

        mu = float(np.mean(pnls))
        sigma = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 1e-12

        # Sharpe
        sharpe = (mu / sigma * np.sqrt(self.annualisation)) if sigma > 1e-12 else 0.0

        # Sortino (downside deviation only)
        downside = pnls[pnls < 0]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-12
        sortino = (mu / downside_std * np.sqrt(self.annualisation)) if downside_std > 1e-12 else 0.0

        # Max drawdown
        cum_pnl = np.cumsum(pnls)
        mdd = PnLAttributor.max_drawdown(cum_pnl)

        # Hit rate
        hit_rate = float(np.mean(pnls > 0))

        # Annual return (simple scalar: average daily × 252)
        annual_return = mu * self.annualisation

        # Calmar
        calmar = PnLAttributor.calmar_ratio(annual_return, mdd)

        metrics = {
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "max_drawdown": round(mdd, 4),
            "hit_rate": round(hit_rate, 4),
            "calmar": round(calmar, 4),
            "annual_return": round(annual_return, 2),
            "total_pnl": round(float(np.sum(pnls)), 2),
        }
        logger.info("Performance metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Results display
    # ------------------------------------------------------------------

    def print_results(self, results: pd.DataFrame, metrics: dict) -> None:
        """
        Print a formatted results table to the terminal.

        Uses *rich* if available, otherwise falls back to plain text.

        Parameters
        ----------
        results : pd.DataFrame
            Output of :meth:`run`.
        metrics : dict
            Output of :meth:`compute_performance_metrics`.
        """
        if _RICH_AVAILABLE:
            self._print_rich(results, metrics)
        else:
            self._print_plain(results, metrics)

    def _print_rich(self, results: pd.DataFrame, metrics: dict) -> None:
        console = Console()

        # ---- Performance metrics table ----
        perf_table = Table(title="Performance Metrics", show_header=True, header_style="bold magenta")
        perf_table.add_column("Metric", style="cyan", no_wrap=True)
        perf_table.add_column("Value", justify="right")

        metric_labels = {
            "sharpe": "Sharpe Ratio",
            "sortino": "Sortino Ratio",
            "max_drawdown": "Max Drawdown ($)",
            "hit_rate": "Hit Rate",
            "calmar": "Calmar Ratio",
            "annual_return": "Annual Return ($)",
            "total_pnl": "Total P&L ($)",
        }
        for key, label in metric_labels.items():
            perf_table.add_row(label, f"{metrics.get(key, 0):.4f}")

        console.print(perf_table)

        # ---- Daily results table (last 10 rows) ----
        daily_table = Table(
            title=f"Daily Results (last {min(10, len(results))} rows)",
            show_header=True,
            header_style="bold blue",
        )
        for col in ["date", "composite_signal", "atm_iv", "daily_pnl", "cumulative_pnl"]:
            daily_table.add_column(str(col), justify="right")

        tail = results.tail(10)
        for _, row in tail.iterrows():
            daily_table.add_row(
                str(row.get("date", "")),
                f"{row.get('composite_signal', 0):.4f}",
                f"{row.get('atm_iv', 0):.4f}",
                f"{row.get('daily_pnl', 0):.2f}",
                f"{row.get('cumulative_pnl', 0):.2f}",
            )

        console.print(daily_table)

    def _print_plain(self, results: pd.DataFrame, metrics: dict) -> None:
        print("\n=== Performance Metrics ===")
        for key, val in metrics.items():
            print(f"  {key:20s}: {val}")
        print("\n=== Daily Results (last 10 rows) ===")
        cols = ["date", "composite_signal", "atm_iv", "daily_pnl", "cumulative_pnl"]
        available = [c for c in cols if c in results.columns]
        print(results[available].tail(10).to_string(index=False))
