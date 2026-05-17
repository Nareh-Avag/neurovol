"""
Walk-Forward Cross-Validation Runner
=====================================
Implements time-series aware walk-forward validation with embargo periods
to prevent data leakage.  Each fold trains on a strictly earlier window,
applies an embargo gap, validates on a subsequent window, and tests on the
held-out tail.

Usage (CLI)
-----------
    python -m training.walk_forward --config configs/spx.yaml

Usage (API)
-----------
    validator = WalkForwardValidator(config)
    results   = validator.run(surface_tensors, scalar_features, dates)
    summary   = validator.aggregate_results(results)
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

import pytorch_lightning as pl

logger = logging.getLogger(__name__)

# Optional rich — fail gracefully
try:
    from rich.console import Console
    from rich.table import Table
    _RICH_AVAILABLE = True
    _console = Console()
except ImportError:
    _RICH_AVAILABLE = False
    _console = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Fold-split logic
# ---------------------------------------------------------------------------

class WalkForwardValidator:
    """
    Orchestrates walk-forward cross-validation for the volatility surface model.

    Parameters
    ----------
    config : dict
        Expected keys:
          walk_forward.n_folds        – number of folds (default 5)
          walk_forward.train_days     – training window in days (default 504)
          walk_forward.val_days       – validation window in days (default 63)
          walk_forward.test_days      – test window in days (default 21)
          walk_forward.embargo_days   – gap between train-end and val-start (default 5)
          walk_forward.step_days      – step between fold starts (default 63)
          training.*                  – passed to VolSurfaceTrainer
          datamodule.*                – passed to VolSurfaceDataModule
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        wf_cfg = config.get("walk_forward", {})

        self.n_folds: int = int(wf_cfg.get("n_folds", 5))
        self.train_days: int = int(wf_cfg.get("train_days", 504))
        self.val_days: int = int(wf_cfg.get("val_days", 63))
        self.test_days: int = int(wf_cfg.get("test_days", 21))
        self.embargo_days: int = int(wf_cfg.get("embargo_days", 5))
        self.step_days: int = int(wf_cfg.get("step_days", 63))

        self.output_dir: str = config.get("output_dir", "logs/walk_forward")

        logger.info(
            "WalkForwardValidator: n_folds=%d  train=%d  val=%d  test=%d  embargo=%d",
            self.n_folds, self.train_days, self.val_days, self.test_days, self.embargo_days,
        )

    # ------------------------------------------------------------------
    # Fold generation
    # ------------------------------------------------------------------

    def get_fold_splits(self, dates: list) -> list[dict]:
        """
        Generate fold boundary dictionaries.

        Parameters
        ----------
        dates : list
            Sorted list of date objects (or ISO-8601 strings) for all samples.

        Returns
        -------
        list[dict]
            Each dict has keys:
              train_start, train_end, val_start, val_end, test_start, test_end
              (values are date objects)
              train_idx, val_idx, test_idx  (np.ndarray of integer indices)
        """
        # Normalize to date objects
        parsed = []
        for d in dates:
            if isinstance(d, str):
                parsed.append(datetime.fromisoformat(d).date())
            elif hasattr(d, "date"):
                parsed.append(d.date())
            else:
                parsed.append(d)

        date_arr = np.array(parsed)
        n = len(date_arr)

        folds: list[dict] = []
        # Determine anchor of the first fold so all n_folds fit
        fold_anchor = self.train_days
        for fold_idx in range(self.n_folds):
            start_pos = fold_anchor + fold_idx * self.step_days
            train_end_pos = start_pos
            val_start_pos = train_end_pos + self.embargo_days
            val_end_pos = val_start_pos + self.val_days
            test_start_pos = val_end_pos
            test_end_pos = test_start_pos + self.test_days

            if test_end_pos > n:
                logger.warning(
                    "Fold %d would exceed dataset length (%d); truncating.", fold_idx, n
                )
                test_end_pos = n
                if test_start_pos >= n:
                    logger.warning("Fold %d has no test data; skipping.", fold_idx)
                    break

            train_start_pos = max(0, start_pos - self.train_days)

            train_idx = np.arange(train_start_pos, train_end_pos)
            val_idx = np.arange(val_start_pos, val_end_pos)
            test_idx = np.arange(test_start_pos, test_end_pos)

            fold = {
                "fold": fold_idx,
                "train_start": date_arr[train_start_pos],
                "train_end": date_arr[train_end_pos - 1],
                "val_start": date_arr[val_start_pos] if val_start_pos < n else None,
                "val_end": date_arr[min(val_end_pos, n) - 1],
                "test_start": date_arr[test_start_pos] if test_start_pos < n else None,
                "test_end": date_arr[test_end_pos - 1],
                "train_idx": train_idx,
                "val_idx": val_idx,
                "test_idx": test_idx,
            }
            folds.append(fold)
            logger.info(
                "Fold %d: train=[%s..%s] embargo=%dd val=[%s..%s] test=[%s..%s]",
                fold_idx,
                fold["train_start"], fold["train_end"],
                self.embargo_days,
                fold["val_start"], fold["val_end"],
                fold["test_start"], fold["test_end"],
            )

        return folds

    # ------------------------------------------------------------------
    # Training helper (one fold)
    # ------------------------------------------------------------------

    def _train_fold(
        self,
        fold: dict,
        surface_tensors: np.ndarray,
        scalar_features: np.ndarray,
        targets: np.ndarray,
    ) -> dict:
        """Train a fresh model on a single fold and return test metrics."""
        from training.datamodule import VolSurfaceDataModule, VolSurfaceDataset
        from training.trainer import VolSurfaceTrainer
        import tempfile, os

        fold_idx = fold["fold"]
        logger.info("=== Training fold %d ===", fold_idx)

        train_idx = fold["train_idx"]
        val_idx = fold["val_idx"]
        test_idx = fold["test_idx"]

        # Build datasets from index slices
        train_ds = VolSurfaceDataset(
            surface_tensors[train_idx],
            scalar_features[train_idx],
            targets[train_idx],
        )
        val_ds = VolSurfaceDataset(
            surface_tensors[val_idx],
            scalar_features[val_idx],
            targets[val_idx],
        )
        test_ds = VolSurfaceDataset(
            surface_tensors[test_idx],
            scalar_features[test_idx],
            targets[test_idx],
        )

        from torch.utils.data import DataLoader

        batch_size: int = self.config.get("batch_size", 32)
        num_workers: int = self.config.get("num_workers", 0)

        def make_loader(ds, shuffle):
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=False)

        # Trainer lightning module
        lightning_module = VolSurfaceTrainer(self.config)

        # Callbacks
        from training.callbacks import EarlyStoppingCallback, ModelCheckpointCallback
        fold_ckpt_dir = os.path.join(self.output_dir, f"fold_{fold_idx}", "checkpoints")
        os.makedirs(fold_ckpt_dir, exist_ok=True)

        ckpt_callback = ModelCheckpointCallback(dirpath=fold_ckpt_dir)
        early_stop = EarlyStoppingCallback(self.config)

        max_epochs: int = self.config.get("training", {}).get("max_epochs", 100)

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[ckpt_callback, early_stop],
            enable_progress_bar=False,
            logger=False,
            accelerator="auto",
            devices=1,
        )
        trainer.fit(
            lightning_module,
            train_dataloaders=make_loader(train_ds, shuffle=True),
            val_dataloaders=make_loader(val_ds, shuffle=False),
        )

        # Test
        test_results = trainer.test(
            lightning_module,
            dataloaders=make_loader(test_ds, shuffle=False),
            verbose=False,
        )

        metrics: dict[str, float] = {}
        if test_results:
            metrics.update(test_results[0])

        metrics["fold"] = fold_idx
        metrics["train_size"] = len(train_idx)
        metrics["val_size"] = len(val_idx)
        metrics["test_size"] = len(test_idx)
        metrics["best_model_path"] = ckpt_callback.best_model_path if hasattr(ckpt_callback, "best_model_path") else ""
        logger.info("Fold %d results: %s", fold_idx, metrics)
        return metrics

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        surface_tensors: np.ndarray,
        scalar_features: np.ndarray,
        dates: list,
        targets: Optional[np.ndarray] = None,
    ) -> list[dict]:
        """
        Run walk-forward validation across all folds.

        Parameters
        ----------
        surface_tensors : np.ndarray  (N, T, H, W, C)
        scalar_features : np.ndarray  (N, T, F) or (N, F)
        dates           : list of date/str  length N
        targets         : np.ndarray  (N, horizons, H, W)  — if None, tries to
                          infer from surface_tensors (not recommended).

        Returns
        -------
        list[dict]  – one metrics dict per fold
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        if targets is None:
            raise ValueError("targets must be provided to walk_forward.run()")

        folds = self.get_fold_splits(dates)
        fold_results: list[dict] = []

        for fold in folds:
            result = self._train_fold(fold, surface_tensors, scalar_features, targets)
            fold_results.append(result)

        # Save results
        results_df = pd.DataFrame(fold_results)
        results_path = os.path.join(self.output_dir, "fold_results.csv")
        results_df.to_csv(results_path, index=False)
        logger.info("Fold results saved to %s", results_path)

        return fold_results

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate_results(self, fold_results: list[dict]) -> dict:
        """
        Compute mean and std of numeric metrics across folds.

        Parameters
        ----------
        fold_results : list[dict]  – output of run()

        Returns
        -------
        dict  – keys: "{metric}_mean", "{metric}_std" for each numeric metric
        """
        if not fold_results:
            return {}

        df = pd.DataFrame(fold_results)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude fold index and size counters from mean/std
        skip = {"fold", "train_size", "val_size", "test_size"}
        metric_cols = [c for c in numeric_cols if c not in skip]

        summary: dict[str, float] = {}
        for col in metric_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                summary[f"{col}_mean"] = float(vals.mean())
                summary[f"{col}_std"] = float(vals.std(ddof=0))

        self._print_summary_table(df, summary, metric_cols)
        return summary

    # ------------------------------------------------------------------
    # Rich display
    # ------------------------------------------------------------------

    def _print_summary_table(
        self,
        df: pd.DataFrame,
        summary: dict,
        metric_cols: list[str],
    ) -> None:
        """Print a rich summary table of fold results and aggregated stats."""
        if _RICH_AVAILABLE and _console is not None:
            # Per-fold table
            fold_table = Table(title="Walk-Forward Fold Results", show_lines=True)
            fold_table.add_column("Fold", style="bold cyan", justify="right")
            for col in metric_cols:
                fold_table.add_column(col, justify="right")

            for _, row in df.iterrows():
                vals = [str(row.get("fold", "?"))]
                for col in metric_cols:
                    v = row.get(col, float("nan"))
                    vals.append(f"{v:.5f}" if isinstance(v, float) else str(v))
                fold_table.add_row(*vals)

            _console.print(fold_table)

            # Summary table
            summ_table = Table(title="Aggregated Metrics (mean ± std)", show_lines=True)
            summ_table.add_column("Metric", style="bold green")
            summ_table.add_column("Mean", justify="right")
            summ_table.add_column("Std", justify="right")

            seen: set[str] = set()
            for key in summary:
                if key.endswith("_mean"):
                    base = key[:-5]
                    if base in seen:
                        continue
                    seen.add(base)
                    mean_val = summary.get(f"{base}_mean", float("nan"))
                    std_val = summary.get(f"{base}_std", float("nan"))
                    summ_table.add_row(base, f"{mean_val:.5f}", f"{std_val:.5f}")

            _console.print(summ_table)
        else:
            # Fallback plain-text output
            print("\n=== Walk-Forward Results ===")
            print(df[["fold"] + metric_cols].to_string(index=False))
            print("\n=== Aggregated Stats ===")
            for key, val in summary.items():
                print(f"  {key}: {val:.5f}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Walk-Forward Validation Runner")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--data-dir", default="data/tensors",
                        help="Directory with pre-processed numpy arrays")
    return parser.parse_args()


def main():
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    args = _parse_args()
    with open(args.config, "r") as fh:
        config = yaml.safe_load(fh)

    data_dir = args.data_dir

    logger.info("Loading data from %s", data_dir)
    surfaces = np.load(f"{data_dir}/surfaces.npy", allow_pickle=True)
    scalars = np.load(f"{data_dir}/scalars.npy", allow_pickle=True)
    targets = np.load(f"{data_dir}/targets.npy", allow_pickle=True)
    dates = np.load(f"{data_dir}/dates.npy", allow_pickle=True).tolist()

    validator = WalkForwardValidator(config)
    fold_results = validator.run(surfaces, scalars, dates, targets=targets)
    summary = validator.aggregate_results(fold_results)
    logger.info("Walk-forward complete. Summary: %s", summary)


if __name__ == "__main__":
    main()



