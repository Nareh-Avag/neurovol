"""
Regime analysis for the Neural Volatility Surface Forecasting System.

Classifies market regimes using a Hidden Markov Model (or K-Means as fallback)
and computes regime-conditional surface statistics.
"""

import logging
from typing import Any

import numpy as np
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Try to import hmmlearn; fall back to sklearn KMeans
try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore[import]

    _HMM_AVAILABLE = True
    logger.info("hmmlearn available — using GaussianHMM for regime classification.")
except ImportError:
    _HMM_AVAILABLE = False
    from sklearn.cluster import KMeans

    logger.warning(
        "hmmlearn not available — falling back to sklearn KMeans for regime "
        "classification.  Install hmmlearn for HMM-based regimes."
    )

# Regime label ordering (lowest mean IV → highest)
_REGIME_LABELS = ["low_vol", "normal", "elevated", "crisis"]


class RegimeClassifier:
    """
    Classifies market regimes from feature vectors.

    Uses a Gaussian HMM when hmmlearn is installed, otherwise falls back to
    K-Means clustering (no temporal dependency).

    Parameters
    ----------
    n_regimes : int
        Number of distinct regimes.  Default 4.
    """

    def __init__(self, n_regimes: int = 4) -> None:
        self.n_regimes = n_regimes
        self._model: Any = None
        self._regime_label_map: dict[int, str] = {}
        logger.info("RegimeClassifier initialised with n_regimes=%d", n_regimes)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, features: np.ndarray) -> "RegimeClassifier":
        """
        Fit the regime model on historical feature vectors.

        Parameters
        ----------
        features : np.ndarray
            2-D array of shape (T, D) where T is the number of time steps and
            D is the feature dimension.

        Returns
        -------
        self
        """
        if features.ndim != 2:
            raise ValueError(
                f"features must be 2-D (T, D); got shape {features.shape}."
            )

        if _HMM_AVAILABLE:
            model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=200,
                random_state=42,
                verbose=False,
            )
            model.fit(features)
            self._model = model
            logger.info("GaussianHMM fitted on %d observations.", features.shape[0])
        else:
            model = KMeans(
                n_clusters=self.n_regimes,
                random_state=42,
                n_init=10,
            )
            model.fit(features)
            self._model = model
            logger.info("KMeans fitted on %d observations.", features.shape[0])

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_regime(self, features: np.ndarray) -> np.ndarray:
        """
        Assign regime labels to a sequence of feature vectors.

        Parameters
        ----------
        features : np.ndarray
            2-D array of shape (T, D).

        Returns
        -------
        np.ndarray
            Integer regime labels of shape (T,), in [0, n_regimes).
        """
        self._check_fitted()
        if _HMM_AVAILABLE:
            labels = self._model.predict(features)
        else:
            labels = self._model.predict(features)
        return np.asarray(labels, dtype=int)

    # ------------------------------------------------------------------
    # Label mapping
    # ------------------------------------------------------------------

    def label_regimes(
        self,
        atm_iv_series: np.ndarray,
        regime_labels: np.ndarray,
    ) -> dict[int, str]:
        """
        Map integer regime indices to human-readable labels.

        Regimes are ordered by their mean ATM IV and labelled accordingly:
        lowest → "low_vol", ..., highest → "crisis".

        Parameters
        ----------
        atm_iv_series : np.ndarray
            1-D array of ATM IV values aligned with regime_labels.
        regime_labels : np.ndarray
            Integer regime labels from :meth:`predict_regime`.

        Returns
        -------
        dict[int, str]
            Maps regime index → label string.
        """
        atm_iv = np.asarray(atm_iv_series, dtype=float)
        labels = np.asarray(regime_labels, dtype=int)

        # Compute mean IV per regime
        regime_means: dict[int, float] = {}
        for r in range(self.n_regimes):
            mask = labels == r
            regime_means[r] = float(np.mean(atm_iv[mask])) if mask.any() else 0.0

        # Sort regimes by mean IV
        sorted_regimes = sorted(regime_means.keys(), key=lambda r: regime_means[r])

        # Build label map (cap at available label names)
        available_labels = _REGIME_LABELS[:self.n_regimes]
        if len(available_labels) < self.n_regimes:
            extra = [f"regime_{i}" for i in range(len(available_labels), self.n_regimes)]
            available_labels = available_labels + extra

        self._regime_label_map = {
            sorted_regimes[i]: available_labels[i]
            for i in range(self.n_regimes)
        }
        logger.info(
            "Regime label map: %s (mean IVs: %s)",
            self._regime_label_map,
            {r: f"{v:.4f}" for r, v in regime_means.items()},
        )
        return self._regime_label_map

    # ------------------------------------------------------------------
    # Conditional surface statistics
    # ------------------------------------------------------------------

    def regime_conditional_surface_stats(
        self,
        surfaces: np.ndarray,
        regime_labels: np.ndarray,
    ) -> dict[str, Any]:
        """
        Compute mean (and std) IV surface for each regime.

        Parameters
        ----------
        surfaces : np.ndarray
            Array of shape (T, H, W) of IV surfaces.
        regime_labels : np.ndarray
            Integer regime labels of shape (T,).

        Returns
        -------
        dict[str, Any]
            Keys are regime label strings (or ints if label_regimes not called).
            Each value is a dict: {"mean_surface": np.ndarray (H,W),
                                    "std_surface": np.ndarray (H,W),
                                    "count": int}.
        """
        surfaces = np.asarray(surfaces, dtype=float)
        labels = np.asarray(regime_labels, dtype=int)

        stats: dict[str, Any] = {}
        for r in range(self.n_regimes):
            mask = labels == r
            regime_surfaces = surfaces[mask]
            count = int(mask.sum())

            if count == 0:
                logger.warning("Regime %d has no observations — skipping.", r)
                continue

            mean_surface = regime_surfaces.mean(axis=0)
            std_surface = regime_surfaces.std(axis=0) if count > 1 else np.zeros_like(mean_surface)

            # Use mapped label if available
            key: str = self._regime_label_map.get(r, str(r))
            stats[key] = {
                "mean_surface": mean_surface,
                "std_surface": std_surface,
                "count": count,
            }
            logger.debug("Regime '%s': count=%d mean_atm_iv=%.4f", key, count, float(mean_surface.mean()))

        return stats

    # ------------------------------------------------------------------
    # Visualisation helper
    # ------------------------------------------------------------------

    def plot_regime_timeline(
        self,
        dates: list,
        regime_labels: np.ndarray,
        atm_iv_series: np.ndarray | None = None,
    ) -> go.Figure:
        """
        Plot regime labels over time, optionally overlaid with ATM IV.

        Parameters
        ----------
        dates : list
            Sequence of date labels aligned with regime_labels.
        regime_labels : np.ndarray
            Integer regime labels.
        atm_iv_series : np.ndarray or None
            Optional ATM IV series for overlay.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        labels = np.asarray(regime_labels, dtype=int)
        colour_map = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]  # green/blue/amber/red

        fig = go.Figure()

        # Regime as step function
        regime_colours = [colour_map[min(r, len(colour_map) - 1)] for r in labels]
        for r in range(self.n_regimes):
            mask = labels == r
            label_name = self._regime_label_map.get(r, f"Regime {r}")
            date_subset = [d for d, m in zip(dates, mask) if m]
            regime_subset = [r] * len(date_subset)
            fig.add_trace(
                go.Scatter(
                    x=date_subset,
                    y=regime_subset,
                    mode="markers",
                    name=label_name,
                    marker=dict(color=colour_map[min(r, len(colour_map) - 1)], size=6),
                )
            )

        if atm_iv_series is not None:
            fig.add_trace(
                go.Scatter(
                    x=list(dates),
                    y=list(atm_iv_series),
                    name="ATM IV",
                    mode="lines",
                    line=dict(color="black", width=1.5),
                    yaxis="y2",
                )
            )

        fig.update_layout(
            title="Market Regime Timeline",
            xaxis_title="Date",
            yaxis=dict(title="Regime Index", dtick=1),
            yaxis2=dict(title="ATM IV", overlaying="y", side="right") if atm_iv_series is not None else {},
            template="plotly_white",
            legend=dict(x=0.01, y=0.99),
        )
        return fig

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("RegimeClassifier is not fitted yet. Call fit() first.")
