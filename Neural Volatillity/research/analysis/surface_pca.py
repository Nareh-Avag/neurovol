"""
Surface PCA analysis for the Neural Volatility Surface Forecasting System.

Fits a PCA model on flattened IV-surface tensors and provides reconstruction,
explained-variance visualisation, and factor interpretation.
"""

import logging
from typing import Any

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class SurfacePCA:
    """
    Principal Component Analysis of implied-volatility surfaces.

    Surfaces are represented as 2-D grids of shape (H, W) where H is the
    number of expiry nodes and W is the number of moneyness/strike nodes.
    Each surface is flattened to a 1-D vector of length H×W before fitting.

    Parameters
    ----------
    n_components : int
        Number of principal components to retain.  Default 10.
    """

    def __init__(self, n_components: int = 10) -> None:
        self.n_components = n_components
        self._pca: PCA | None = None
        self._surface_shape: tuple[int, int] | None = None
        logger.info("SurfacePCA initialised with n_components=%d", n_components)

    # ------------------------------------------------------------------
    # Fit / transform / reconstruct
    # ------------------------------------------------------------------

    def fit(self, surface_tensors: np.ndarray) -> "SurfacePCA":
        """
        Fit PCA on a batch of IV surfaces.

        Parameters
        ----------
        surface_tensors : np.ndarray
            Array of shape (N, H, W) – N surfaces each of spatial size H×W.

        Returns
        -------
        self
        """
        if surface_tensors.ndim != 3:
            raise ValueError(
                f"surface_tensors must be 3-D (N, H, W); got shape {surface_tensors.shape}."
            )
        n, h, w = surface_tensors.shape
        self._surface_shape = (h, w)

        X: np.ndarray = surface_tensors.reshape(n, h * w)
        n_comp = min(self.n_components, n, h * w)
        self._pca = PCA(n_components=n_comp, random_state=42)
        self._pca.fit(X)

        logger.info(
            "SurfacePCA fitted on %d surfaces (H=%d, W=%d). "
            "Explained variance ratio (top 3): %s",
            n, h, w,
            self._pca.explained_variance_ratio_[:3].round(4).tolist(),
        )
        return self

    def transform(self, surface_tensors: np.ndarray) -> np.ndarray:
        """
        Project surfaces into PC score space.

        Parameters
        ----------
        surface_tensors : np.ndarray
            Array of shape (N, H, W).

        Returns
        -------
        np.ndarray
            PC scores of shape (N, n_components).
        """
        self._check_fitted()
        n, h, w = surface_tensors.shape
        X = surface_tensors.reshape(n, h * w)
        return self._pca.transform(X)  # type: ignore[union-attr]

    def reconstruct(self, pc_scores: np.ndarray) -> np.ndarray:
        """
        Reconstruct IV surfaces from PC scores.

        Parameters
        ----------
        pc_scores : np.ndarray
            Array of shape (N, n_components).

        Returns
        -------
        np.ndarray
            Reconstructed surfaces of shape (N, H, W).
        """
        self._check_fitted()
        h, w = self._surface_shape  # type: ignore[misc]
        X_recon = self._pca.inverse_transform(pc_scores)  # type: ignore[union-attr]
        return X_recon.reshape(-1, h, w)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def explained_variance_plot(self) -> go.Figure:
        """
        Bar chart of explained variance ratio per principal component.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        self._check_fitted()
        evr = self._pca.explained_variance_ratio_  # type: ignore[union-attr]
        cumulative = np.cumsum(evr)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[f"PC{i + 1}" for i in range(len(evr))],
                y=evr,
                name="Individual",
                marker_color="royalblue",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[f"PC{i + 1}" for i in range(len(evr))],
                y=cumulative,
                name="Cumulative",
                mode="lines+markers",
                line=dict(color="firebrick", width=2),
                yaxis="y2",
            )
        )
        fig.update_layout(
            title="PCA Explained Variance Ratio",
            xaxis_title="Principal Component",
            yaxis=dict(title="Explained Variance Ratio", tickformat=".1%"),
            yaxis2=dict(
                title="Cumulative Explained Variance",
                overlaying="y",
                side="right",
                tickformat=".1%",
                range=[0, 1.05],
            ),
            legend=dict(x=0.6, y=0.2),
            template="plotly_white",
        )
        return fig

    def factor_surface_plot(self, component_idx: int) -> go.Figure:
        """
        Heatmap of the PC loadings for a given component, reshaped as (H, W).

        Parameters
        ----------
        component_idx : int
            Zero-based index of the principal component.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        self._check_fitted()
        pca: PCA = self._pca  # type: ignore[assignment]
        if component_idx >= pca.n_components_:
            raise ValueError(
                f"component_idx={component_idx} out of range "
                f"(n_components={pca.n_components_})."
            )
        h, w = self._surface_shape  # type: ignore[misc]
        loadings = pca.components_[component_idx].reshape(h, w)

        fig = go.Figure(
            data=go.Heatmap(
                z=loadings,
                colorscale="RdBu",
                zmid=0,
                colorbar=dict(title="Loading"),
            )
        )
        fig.update_layout(
            title=f"PC{component_idx + 1} Loadings (surface view)",
            xaxis_title="Strike / Moneyness Node",
            yaxis_title="Expiry Node",
            template="plotly_white",
        )
        return fig

    # ------------------------------------------------------------------
    # Factor interpretation
    # ------------------------------------------------------------------

    def interpret_factors(self) -> dict[str, Any]:
        """
        Heuristically label the first three principal components.

        Methodology
        -----------
        * PC1 (Level): dominant component; loadings should be same-signed
          across the surface → mean IV shift.
        * PC2 (Skew / tilt): loadings change sign across the strike axis →
          left/right tilt.
        * PC3 (Curvature / smile): loadings show a U-shape across strikes →
          smile curvature.

        The interpretation is based on the sign-pattern of the component
        vector; it is heuristic and should be validated graphically.

        Returns
        -------
        dict[str, Any]
            {
              "PC1": {"label": "level", "mean_loading": float, ...},
              "PC2": {"label": "skew",  ...},
              "PC3": {"label": "curvature", ...},
            }
        """
        self._check_fitted()
        pca: PCA = self._pca  # type: ignore[assignment]
        h, w = self._surface_shape  # type: ignore[misc]
        result: dict[str, Any] = {}

        label_map = {0: "level", 1: "skew", 2: "curvature"}
        for i in range(min(3, pca.n_components_)):
            comp = pca.components_[i]
            comp_2d = comp.reshape(h, w)
            # Strike-axis sign changes (columns)
            col_means = comp_2d.mean(axis=0)
            sign_changes = int(np.sum(np.diff(np.sign(col_means)) != 0))

            result[f"PC{i + 1}"] = {
                "label": label_map[i],
                "explained_variance_ratio": float(pca.explained_variance_ratio_[i]),
                "mean_loading": float(np.mean(comp)),
                "strike_axis_sign_changes": sign_changes,
                "component_norm": float(np.linalg.norm(comp)),
            }

        logger.info("Factor interpretation: %s", result)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._pca is None:
            raise RuntimeError("SurfacePCA is not fitted yet. Call fit() first.")
