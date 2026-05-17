"""
Signal generator for the Neural Volatility Surface Forecasting System.

Produces trading signals in [-1, 1] from forecast vs current surface statistics.
Positive signals denote a recommendation to sell volatility / benefit from the
direction indicated by the signal name.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Converts volatility-surface forecasts and current surface statistics into
    normalised trading signals.

    Parameters
    ----------
    config : dict
        Accepted keys:
          - vol_carry_threshold (float): minimum |(IV - RV)| / IV to generate a
            non-zero carry signal.  Default 0.02.
          - skew_scale (float): denominator used to clip the skew signal.
            Default 0.05.
          - term_structure_scale (float): denominator used to clip the term-
            structure signal.  Default 0.03.
          - inversion_prob_threshold (float): global default probability
            threshold for the inversion signal.  Default 0.60.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.vol_carry_threshold: float = float(config.get("vol_carry_threshold", 0.02))
        self.skew_scale: float = float(config.get("skew_scale", 0.05))
        self.term_structure_scale: float = float(config.get("term_structure_scale", 0.03))
        self.inversion_prob_threshold: float = float(
            config.get("inversion_prob_threshold", 0.60)
        )
        logger.info(
            "SignalGenerator initialised with config: %s", config
        )

    # ------------------------------------------------------------------
    # Individual signal methods
    # ------------------------------------------------------------------

    def vol_carry_signal(self, atm_iv: float, rv_21d: float) -> float:
        """
        Volatility carry signal.

        Compares ATM implied volatility to a 21-day realised volatility.
        A positive value means IV > RV, so selling vol is expected to be
        profitable (positive carry).

        Returns
        -------
        float
            Signal in [-1, 1].  Positive → sell vol, negative → buy vol.
        """
        if atm_iv <= 0:
            logger.warning("atm_iv must be positive; received %.6f – returning 0.", atm_iv)
            return 0.0

        carry: float = (atm_iv - rv_21d) / atm_iv
        # Apply a dead-band around zero
        if abs(carry) < self.vol_carry_threshold:
            signal = 0.0
        else:
            signal = float(np.clip(carry, -1.0, 1.0))

        logger.debug(
            "vol_carry_signal: atm_iv=%.4f rv_21d=%.4f carry=%.4f signal=%.4f",
            atm_iv, rv_21d, carry, signal,
        )
        return signal

    def skew_signal(self, current_skew: float, forecast_skew: float) -> float:
        """
        Skew signal.

        Positive value means the forecast predicts a steeper (more negative)
        skew than the current skew, suggesting the market will become more
        left-skewed.

        Parameters
        ----------
        current_skew : float
            Current 25Δ put-call skew (put IV − call IV), typically negative.
        forecast_skew : float
            Model forecast of the same metric.

        Returns
        -------
        float
            Signal in [-1, 1].  Positive → forecast steepening.
        """
        delta: float = forecast_skew - current_skew
        signal: float = float(np.clip(delta / self.skew_scale, -1.0, 1.0))
        logger.debug(
            "skew_signal: current=%.4f forecast=%.4f delta=%.4f signal=%.4f",
            current_skew, forecast_skew, delta, signal,
        )
        return signal

    def term_structure_signal(
        self, current_slope: float, forecast_slope: float
    ) -> float:
        """
        Term-structure slope signal.

        Positive value means the forecast predicts a steeper upward-sloping
        term structure (longer dated IV > shorter dated IV).

        Parameters
        ----------
        current_slope : float
            Current slope: IV(3M) − IV(1M), in volatility points.
        forecast_slope : float
            Model forecast of the same metric.

        Returns
        -------
        float
            Signal in [-1, 1].
        """
        delta: float = forecast_slope - current_slope
        signal: float = float(np.clip(delta / self.term_structure_scale, -1.0, 1.0))
        logger.debug(
            "term_structure_signal: current=%.4f forecast=%.4f delta=%.4f signal=%.4f",
            current_slope, forecast_slope, delta, signal,
        )
        return signal

    def inversion_signal(
        self, inversion_prob: float, threshold: float = 0.60
    ) -> int:
        """
        Term-structure inversion signal.

        Parameters
        ----------
        inversion_prob : float
            Model-estimated probability (0–1) that the term structure will
            invert (short-dated IV exceeding long-dated IV).
        threshold : float
            Probability threshold above which the signal fires.  Defaults to
            0.60 but can be overridden per call.

        Returns
        -------
        int
            1 if inversion is likely (prob ≥ threshold), 0 otherwise.
        """
        effective_threshold = threshold if threshold is not None else self.inversion_prob_threshold
        signal: int = 1 if inversion_prob >= effective_threshold else 0
        logger.debug(
            "inversion_signal: prob=%.4f threshold=%.4f signal=%d",
            inversion_prob, effective_threshold, signal,
        )
        return signal

    # ------------------------------------------------------------------
    # Aggregate signal method
    # ------------------------------------------------------------------

    def generate_all(
        self,
        forecast: dict,
        current_surface_stats: dict,
    ) -> dict[str, Any]:
        """
        Generate all signals and return them as a dictionary.

        Parameters
        ----------
        forecast : dict
            Expected keys:
              - atm_iv        (float): forecast ATM implied vol
              - skew          (float): forecast 25Δ skew
              - term_slope    (float): forecast term-structure slope IV(3M)-IV(1M)
              - inversion_prob(float): forecast probability of term inversion
        current_surface_stats : dict
            Expected keys:
              - atm_iv        (float): current ATM implied vol
              - rv_21d        (float): 21-day realised vol
              - skew          (float): current 25Δ skew
              - term_slope    (float): current term-structure slope

        Returns
        -------
        dict
            {
              "vol_carry": float,
              "skew": float,
              "term_structure": float,
              "inversion": int,
              "composite": float,
            }
            ``composite`` is a simple equal-weight average of the numeric
            signals, clipped to [-1, 1].
        """
        # --- Extract fields with sensible defaults --------------------------------
        current_atm_iv: float = float(current_surface_stats.get("atm_iv", 0.20))
        rv_21d: float = float(current_surface_stats.get("rv_21d", 0.18))
        current_skew: float = float(current_surface_stats.get("skew", -0.05))
        current_slope: float = float(current_surface_stats.get("term_slope", 0.02))

        forecast_atm_iv: float = float(forecast.get("atm_iv", current_atm_iv))
        forecast_skew: float = float(forecast.get("skew", current_skew))
        forecast_slope: float = float(forecast.get("term_slope", current_slope))
        inversion_prob: float = float(forecast.get("inversion_prob", 0.0))

        # --- Compute individual signals ------------------------------------------
        vc = self.vol_carry_signal(current_atm_iv, rv_21d)
        sk = self.skew_signal(current_skew, forecast_skew)
        ts = self.term_structure_signal(current_slope, forecast_slope)
        inv = self.inversion_signal(inversion_prob)

        # Composite: equal-weight average of continuous signals
        composite: float = float(np.clip(np.mean([vc, sk, ts]), -1.0, 1.0))

        signals: dict[str, Any] = {
            "vol_carry": vc,
            "skew": sk,
            "term_structure": ts,
            "inversion": inv,
            "composite": composite,
        }
        logger.info("Generated signals: %s", signals)
        return signals
