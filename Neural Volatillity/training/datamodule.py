"""
VolSurface DataModule
=====================
PyTorch Lightning DataModule for loading pre-processed volatility surface
tensors from disk and serving them as train / val / test DataLoaders.

Expected on-disk layout (data_dir)
------------------------------------
  surfaces.npy       – float32 array, shape (N, T, H, W, C)
  scalars.npy        – float32 array, shape (N, T, F)
  targets.npy        – float32 array, shape (N, horizons, H, W)
  dates.npy          – object  array, shape (N,)  – ISO-8601 date strings

Where
  N = number of samples
  T = lookback window (e.g. 60 days)
  H = strike grid size (e.g. 19)
  W = expiry grid size (e.g. 14)
  C = surface channels (e.g. 6)
  F = scalar feature count (e.g. 60)
"""

import logging
import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VolSurfaceDataset(Dataset):
    """In-memory dataset wrapping surface + scalar tensors and targets."""

    def __init__(
        self,
        surfaces: np.ndarray,
        scalars: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        surfaces : np.ndarray  shape (N, T, H, W, C)
        scalars  : np.ndarray  shape (N, T, F)  or  (N, F)
        targets  : np.ndarray  shape (N, horizons, H, W)
        """
        assert len(surfaces) == len(scalars) == len(targets), (
            "surfaces, scalars, and targets must have identical first dimension"
        )
        self.surfaces = torch.from_numpy(surfaces.astype(np.float32))
        self.scalars = torch.from_numpy(scalars.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return len(self.surfaces)

    def __getitem__(self, idx: int) -> dict:
        return {
            "surface_seq": self.surfaces[idx],   # (T, H, W, C)
            "scalar_seq": self.scalars[idx],     # (T, F) or (F,)
            "target": self.targets[idx],          # (horizons, H, W)
        }


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class VolSurfaceDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the Neural Volatility Surface system.

    Parameters
    ----------
    config : dict
        Must contain:
          config["batch_size"]       – int
          config["train_frac"]       – float, fraction for training (e.g. 0.7)
          config["val_frac"]         – float, fraction for validation (e.g. 0.15)
          config["seed"]             – int (optional, default 42)
    data_dir : str
        Directory containing surfaces.npy, scalars.npy, targets.npy, dates.npy
    """

    def __init__(self, config: dict, data_dir: str = "data/tensors") -> None:
        super().__init__()
        self.config = config
        self.data_dir = data_dir

        self.batch_size: int = config.get("batch_size", 32)
        self.train_frac: float = config.get("train_frac", 0.70)
        self.val_frac: float = config.get("val_frac", 0.15)
        self.seed: int = config.get("seed", 42)
        self.num_workers: int = config.get("num_workers", 4)

        # Populated in setup()
        self._train_dataset: Optional[VolSurfaceDataset] = None
        self._val_dataset: Optional[VolSurfaceDataset] = None
        self._test_dataset: Optional[VolSurfaceDataset] = None

        self.save_hyperparameters({"config": config, "data_dir": data_dir})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_array(self, filename: str) -> np.ndarray:
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"VolSurfaceDataModule: required file not found: {path}"
            )
        arr = np.load(path, allow_pickle=True)
        logger.info("Loaded %s  shape=%s  dtype=%s", filename, arr.shape, arr.dtype)
        return arr

    def _split_indices(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return deterministic time-ordered train/val/test indices."""
        train_end = int(n * self.train_frac)
        val_end = train_end + int(n * self.val_frac)

        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, n)

        logger.info(
            "Split: train=%d  val=%d  test=%d  (total=%d)",
            len(train_idx), len(val_idx), len(test_idx), n,
        )
        return train_idx, val_idx, test_idx

    # ------------------------------------------------------------------
    # LightningDataModule interface
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        """Check that required files exist (called once on rank-0)."""
        for fname in ("surfaces.npy", "scalars.npy", "targets.npy"):
            path = os.path.join(self.data_dir, fname)
            if not os.path.exists(path):
                logger.warning(
                    "prepare_data: %s not found at %s — "
                    "generate data before training.",
                    fname, path,
                )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load arrays from disk and build dataset splits.

        Called on every GPU/process by Lightning; guard with already-loaded check.
        """
        if self._train_dataset is not None:
            return  # already set up

        surfaces = self._load_array("surfaces.npy")   # (N, T, H, W, C)
        scalars = self._load_array("scalars.npy")     # (N, T, F) or (N, F)
        targets = self._load_array("targets.npy")     # (N, horizons, H, W)

        n = len(surfaces)
        train_idx, val_idx, test_idx = self._split_indices(n)

        if stage in ("fit", None):
            self._train_dataset = VolSurfaceDataset(
                surfaces[train_idx], scalars[train_idx], targets[train_idx]
            )
            self._val_dataset = VolSurfaceDataset(
                surfaces[val_idx], scalars[val_idx], targets[val_idx]
            )
            logger.info(
                "Setup stage=%s  train=%d  val=%d",
                stage, len(self._train_dataset), len(self._val_dataset),
            )

        if stage in ("test", None):
            self._test_dataset = VolSurfaceDataset(
                surfaces[test_idx], scalars[test_idx], targets[test_idx]
            )
            logger.info(
                "Setup stage=%s  test=%d", stage, len(self._test_dataset)
            )

        if stage == "predict":
            # Re-use test split for prediction
            self._test_dataset = VolSurfaceDataset(
                surfaces[test_idx], scalars[test_idx], targets[test_idx]
            )

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------

    def _make_loader(self, dataset: VolSurfaceDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            raise RuntimeError("Call setup('fit') before train_dataloader()")
        return self._make_loader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            raise RuntimeError("Call setup('fit') before val_dataloader()")
        return self._make_loader(self._val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self._test_dataset is None:
            raise RuntimeError("Call setup('test') before test_dataloader()")
        return self._make_loader(self._test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def train_size(self) -> int:
        return len(self._train_dataset) if self._train_dataset else 0

    @property
    def val_size(self) -> int:
        return len(self._val_dataset) if self._val_dataset else 0

    @property
    def test_size(self) -> int:
        return len(self._test_dataset) if self._test_dataset else 0
