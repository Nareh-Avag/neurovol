"""
VolSurface Lightning Module
============================
PyTorch Lightning LightningModule that wraps the model, loss, optimizer, and
logging for the Neural Volatility Surface Forecasting System.

Expected batch dict keys (produced by VolSurfaceDataset):
    surface_seq  – (B, T, H, W, C)   float32
    scalar_seq   – (B, T, F) or (B, F)  float32
    target       – (B, horizons, H, W)  float32

Output from model forward:
    predictions  – (B, horizons, H, W)
"""

import logging
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Optional MLflow — fail gracefully if not installed
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False
    logger.warning("mlflow not available – MLflow logging disabled")

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root mean squared error over all elements."""
    return torch.sqrt(F.mse_loss(pred, target))


def _horizon_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    horizon_idx: int,
) -> torch.Tensor:
    """RMSE for a single horizon slice."""
    return _rmse(pred[:, horizon_idx, :, :], target[:, horizon_idx, :, :])


def _atm_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    horizon_idx: int,
    atm_strike_idx: int,
) -> torch.Tensor:
    """RMSE at the ATM strike for a single horizon."""
    return _rmse(
        pred[:, horizon_idx, atm_strike_idx, :],
        target[:, horizon_idx, atm_strike_idx, :],
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class VolSurfaceTrainer(pl.LightningModule):
    """
    PyTorch Lightning Module for training the Neural Volatility Surface model.

    Parameters
    ----------
    config : dict
        Keys used:
          model.*             – passed to build_model()
          training.lr         – initial learning rate (default 1e-3)
          training.weight_decay – AdamW weight decay (default 1e-4)
          training.t_max      – CosineAnnealingWarmRestarts T_0 (default 50)
          training.t_mult     – CosineAnnealingWarmRestarts T_mult (default 2)
          training.atm_strike_idx – index of ATM strike in strike grid (default 9)
          training.horizons   – list of horizon_days aligned with target dim 1
          mlflow.experiment   – MLflow experiment name (optional)
          mlflow.run_name     – MLflow run name (optional)
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.save_hyperparameters({"config": config})

        training_cfg = config.get("training", {})
        self.lr: float = float(training_cfg.get("lr", 1e-3))
        self.weight_decay: float = float(training_cfg.get("weight_decay", 1e-4))
        self.t_max: int = int(training_cfg.get("t_max", 50))
        self.t_mult: int = int(training_cfg.get("t_mult", 2))
        self.atm_strike_idx: int = int(training_cfg.get("atm_strike_idx", 9))
        self.horizons: list = training_cfg.get("horizons", [1, 5, 21])

        # Build model -------------------------------------------------------
        from models import build_model
        self.model = build_model(config)
        logger.info("Built model: %s", type(self.model).__name__)

        # Build loss --------------------------------------------------------
        from models.losses import CombinedVolLoss
        loss_cfg = config.get("loss", {})
        self.criterion = CombinedVolLoss(**loss_cfg)
        logger.info("Loss: %s", type(self.criterion).__name__)

        # MLflow run --------------------------------------------------------
        self._mlflow_run_id: Optional[str] = None
        if _MLFLOW_AVAILABLE:
            self._setup_mlflow()

    # ------------------------------------------------------------------
    # MLflow helpers
    # ------------------------------------------------------------------

    def _setup_mlflow(self) -> None:
        mlflow_cfg = self.config.get("mlflow", {})
        experiment_name = mlflow_cfg.get("experiment", "neural_vol_surface")
        try:
            mlflow.set_experiment(experiment_name)
            logger.info("MLflow experiment: %s", experiment_name)
        except Exception as exc:
            logger.warning("MLflow setup failed: %s", exc)

    def _mlflow_log(self, metrics: dict, step: int) -> None:
        if not _MLFLOW_AVAILABLE:
            return
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as exc:
            logger.debug("MLflow log_metrics failed: %s", exc)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        surface_seq: torch.Tensor,
        scalar_seq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        surface_seq : (B, T, H, W, C)
        scalar_seq  : (B, T, F) or (B, F)

        Returns
        -------
        predictions : (B, horizons, H, W)
        """
        return self.model(surface_seq, scalar_seq)

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------

    def _shared_step(self, batch: dict) -> dict:
        surface_seq: torch.Tensor = batch["surface_seq"]
        scalar_seq: torch.Tensor = batch["scalar_seq"]
        target: torch.Tensor = batch["target"]

        pred = self(surface_seq, scalar_seq)
        loss = self.criterion(pred, target)

        metrics: dict[str, torch.Tensor] = {"loss": loss}

        # Per-horizon RMSE
        horizon_days = self.horizons
        for i, h in enumerate(horizon_days):
            if i < pred.shape[1]:
                metrics[f"rmse_{h}d"] = _horizon_rmse(pred, target, i)

        # ATM RMSE for 1-day horizon
        if pred.shape[1] >= 1:
            metrics["rmse_atm_1d"] = _atm_rmse(pred, target, 0, self.atm_strike_idx)

        return metrics

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        metrics = self._shared_step(batch)

        self.log("train/loss", metrics["loss"], on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        horizon_days = self.horizons
        for i, h in enumerate(horizon_days):
            key = f"rmse_{h}d"
            if key in metrics:
                self.log(f"train/{key}", metrics[key], on_step=False, on_epoch=True,
                         sync_dist=True)

        # MLflow logging every 50 steps
        if batch_idx % 50 == 0 and _MLFLOW_AVAILABLE:
            mlflow_metrics = {
                "train_loss": metrics["loss"].item(),
            }
            for i, h in enumerate(horizon_days):
                key = f"rmse_{h}d"
                if key in metrics:
                    mlflow_metrics[f"train_{key}"] = metrics[key].item()
            self._mlflow_log(mlflow_metrics, step=self.global_step)

        return metrics["loss"]

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        metrics = self._shared_step(batch)

        self.log("val/loss", metrics["loss"], on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        horizon_days = self.horizons
        for i, h in enumerate(horizon_days):
            key = f"rmse_{h}d"
            if key in metrics:
                self.log(f"val/{key}", metrics[key], on_step=False, on_epoch=True,
                         prog_bar=(h == 1), sync_dist=True)

        if "rmse_atm_1d" in metrics:
            self.log("val/rmse_atm_1d", metrics["rmse_atm_1d"],
                     on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        # Log aggregated val metrics to MLflow at epoch end
        if _MLFLOW_AVAILABLE and self.trainer is not None:
            logged = self.trainer.logged_metrics
            mlflow_metrics = {}
            for key, val in logged.items():
                if key.startswith("val/"):
                    mlflow_metrics[key.replace("/", "_")] = float(val)
            if mlflow_metrics:
                self._mlflow_log(mlflow_metrics, step=self.current_epoch)

    # ------------------------------------------------------------------
    # Test step
    # ------------------------------------------------------------------

    def test_step(self, batch: dict, batch_idx: int) -> None:
        metrics = self._shared_step(batch)

        self.log("test/loss", metrics["loss"], on_step=False, on_epoch=True,
                 sync_dist=True)

        horizon_days = self.horizons
        for i, h in enumerate(horizon_days):
            key = f"rmse_{h}d"
            if key in metrics:
                self.log(f"test/{key}", metrics[key], on_step=False, on_epoch=True,
                         sync_dist=True)

        if "rmse_atm_1d" in metrics:
            self.log("test/rmse_atm_1d", metrics["rmse_atm_1d"],
                     on_step=False, on_epoch=True, sync_dist=True)

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.t_max,
            T_mult=self.t_mult,
            eta_min=self.lr * 1e-3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/loss",
            },
        }

    # ------------------------------------------------------------------
    # MLflow lifecycle
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:
        if not _MLFLOW_AVAILABLE:
            return
        try:
            run = mlflow.start_run(
                run_name=self.config.get("mlflow", {}).get("run_name"),
                nested=True,
            )
            self._mlflow_run_id = run.info.run_id
            # Log hyperparameters
            flat_params = _flatten_dict(self.config)
            mlflow.log_params({k: v for k, v in flat_params.items() if len(str(v)) < 250})
            logger.info("MLflow run started: %s", self._mlflow_run_id)
        except Exception as exc:
            logger.warning("MLflow on_train_start failed: %s", exc)

    def on_train_end(self) -> None:
        if not _MLFLOW_AVAILABLE:
            return
        try:
            mlflow.end_run()
            logger.info("MLflow run ended: %s", self._mlflow_run_id)
        except Exception as exc:
            logger.debug("MLflow on_train_end failed: %s", exc)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Recursively flatten a nested dict for MLflow param logging."""
    items: list = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
