"""
Airflow DAG: weekly_model_retraining

Runs a weekly walk-forward model retraining cycle and promotes a new
champion if the challenger outperforms on held-out validation data.

Schedule: Sunday at 2 AM UTC.

Task graph
----------
validate_training_data
    └─> run_walk_forward_training
            └─> evaluate_challenger
                    └─> compare_champion_challenger
                            ├─> [if better]  promote_if_better
                            └─> [if worse]   archive_challenger
                                    └─> notify_team  (both branches join here)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default arguments
# ---------------------------------------------------------------------------
DEFAULT_ARGS: dict = {
    "owner": "quant-research",
    "depends_on_past": False,
    "email": ["narehavag@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=4),
}

# XCom keys
XCOM_CHAMPION_METRICS = "champion_metrics"
XCOM_CHALLENGER_METRICS = "challenger_metrics"
XCOM_CHALLENGER_PATH = "challenger_model_path"
XCOM_COMPARISON_RESULT = "comparison_result"

# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def _validate_training_data(**context) -> None:
    """
    Validate the full historical dataset before training begins.

    Checks:
    - Sufficient history (≥ 252 trading days)
    - No gaps larger than 5 business days
    - IV surface tensor shape consistency
    - No NaN / Inf values in feature matrix
    - Target variable distribution sanity

    Raises
    ------
    ValueError
        If any validation gate fails.
    """
    execution_date: str = str(context.get("ds", datetime.utcnow().date()))
    logger.info("Validating training data as of %s", execution_date)

    # --- placeholder ---
    # dataset = data_loader.load_full_history(as_of=execution_date)
    # assert len(dataset) >= 252, "Insufficient history"
    # assert dataset.isnull().sum().sum() == 0, "NaN values detected"
    # assert dataset["atm_iv"].between(0.01, 2.0).all(), "ATM IV out of range"

    logger.info("Training data validation passed.")


def _run_walk_forward_training(**context) -> None:
    """
    Execute walk-forward cross-validation and train the challenger model.

    Walk-forward configuration (from config):
    - Initial training window: 504 days
    - Validation window:       63 days
    - Step size:               21 days
    - Architecture:            ConvLSTM / Transformer (from model registry config)

    The best checkpoint (lowest validation MSE) is saved to the model store.

    XCom push
    ----------
    ``challenger_model_path`` – URI of the saved model artefact.
    """
    ti = context["ti"]
    execution_date: str = str(context.get("ds", datetime.utcnow().date()))
    logger.info("Starting walk-forward training for %s", execution_date)

    # --- placeholder ---
    # trainer = WalkForwardTrainer(config=MODEL_CONFIG)
    # challenger_path = trainer.fit_and_save(as_of=execution_date)

    challenger_path = f"s3://vol-surface-models/challenger/{execution_date}/model.pt"
    ti.xcom_push(key=XCOM_CHALLENGER_PATH, value=challenger_path)
    logger.info("Challenger model saved to %s", challenger_path)


def _evaluate_challenger(**context) -> None:
    """
    Evaluate the challenger model on the held-out test set.

    Metrics computed:
    - Surface RMSE (overall and per-expiry bucket)
    - ATM IV forecast MAE
    - Skew forecast MAE
    - Directional accuracy (sign of IV change)
    - Calibration score (reliability diagram)

    XCom push
    ----------
    ``challenger_metrics`` – dict of evaluation metrics.
    """
    ti = context["ti"]
    challenger_path: str = ti.xcom_pull(
        task_ids="run_walk_forward_training", key=XCOM_CHALLENGER_PATH
    )
    execution_date: str = str(context.get("ds", datetime.utcnow().date()))
    logger.info("Evaluating challenger from %s", challenger_path)

    # --- placeholder ---
    # model = ModelLoader.load(challenger_path)
    # metrics = Evaluator.evaluate(model, test_set)

    challenger_metrics: dict = {
        "surface_rmse": 0.0085,
        "atm_iv_mae": 0.0042,
        "skew_mae": 0.0031,
        "directional_accuracy": 0.612,
        "calibration_score": 0.94,
        "evaluated_at": execution_date,
    }
    ti.xcom_push(key=XCOM_CHALLENGER_METRICS, value=challenger_metrics)
    logger.info("Challenger metrics: %s", challenger_metrics)


def _compare_champion_challenger(**context) -> str:
    """
    Compare challenger metrics against the current champion.

    Returns the task_id of the next branch to execute:
    - ``"promote_if_better"``  if challenger is better
    - ``"archive_challenger"`` if champion is better or equal

    Promotion criterion: challenger surface_rmse < champion surface_rmse
    with a minimum improvement threshold of 2%.
    """
    ti = context["ti"]
    challenger_metrics: dict = ti.xcom_pull(
        task_ids="evaluate_challenger", key=XCOM_CHALLENGER_METRICS
    )

    # --- placeholder: load champion metrics from model registry ---
    # champion_metrics = model_registry.get_champion_metrics()
    champion_metrics: dict = {
        "surface_rmse": 0.0091,
        "atm_iv_mae": 0.0048,
        "skew_mae": 0.0035,
        "directional_accuracy": 0.598,
        "calibration_score": 0.91,
    }
    ti.xcom_push(key=XCOM_CHAMPION_METRICS, value=champion_metrics)

    challenger_rmse: float = float(challenger_metrics.get("surface_rmse", 1.0))
    champion_rmse: float = float(champion_metrics.get("surface_rmse", 1.0))

    min_improvement = 0.02  # 2% threshold
    relative_improvement = (champion_rmse - challenger_rmse) / max(champion_rmse, 1e-12)

    logger.info(
        "Comparison — champion RMSE: %.6f, challenger RMSE: %.6f, "
        "relative improvement: %.2f%%",
        champion_rmse, challenger_rmse, relative_improvement * 100,
    )

    if relative_improvement > min_improvement:
        logger.info("Challenger is better — branching to promote_if_better.")
        return "promote_if_better"
    else:
        logger.info("Champion retained — branching to archive_challenger.")
        return "archive_challenger"


def _promote_if_better(**context) -> None:
    """
    Promote the challenger model to champion in the model registry.

    Steps:
    1. Archive the current champion with its metrics and date.
    2. Register the challenger as the new champion.
    3. Update the production inference endpoint to load the new weights.
    4. Write a promotion audit log entry.
    """
    ti = context["ti"]
    challenger_path: str = ti.xcom_pull(
        task_ids="run_walk_forward_training", key=XCOM_CHALLENGER_PATH
    )
    challenger_metrics: dict = ti.xcom_pull(
        task_ids="evaluate_challenger", key=XCOM_CHALLENGER_METRICS
    )
    execution_date: str = str(context.get("ds", datetime.utcnow().date()))

    logger.info(
        "Promoting challenger at %s to champion (metrics: %s)",
        challenger_path, challenger_metrics,
    )

    # --- placeholder ---
    # model_registry.promote(challenger_path, metrics=challenger_metrics)
    # inference_service.hot_reload(challenger_path)

    logger.info("Promotion complete as of %s", execution_date)


def _archive_challenger(**context) -> None:
    """
    Archive the challenger model without promoting it.

    The model weights and metrics are retained for audit purposes but the
    champion is not changed.
    """
    ti = context["ti"]
    challenger_path: str = ti.xcom_pull(
        task_ids="run_walk_forward_training", key=XCOM_CHALLENGER_PATH
    )
    champion_metrics: dict = ti.xcom_pull(
        task_ids="compare_champion_challenger", key=XCOM_CHAMPION_METRICS
    )
    challenger_metrics: dict = ti.xcom_pull(
        task_ids="evaluate_challenger", key=XCOM_CHALLENGER_METRICS
    )
    logger.info(
        "Archiving challenger at %s. "
        "Champion surface_rmse=%.6f vs challenger surface_rmse=%.6f",
        challenger_path,
        champion_metrics.get("surface_rmse", float("nan")),
        challenger_metrics.get("surface_rmse", float("nan")),
    )

    # --- placeholder ---
    # model_registry.archive(challenger_path, reason="below threshold", metrics=challenger_metrics)

    logger.info("Challenger archived.")


def _notify_team(**context) -> None:
    """
    Send a Slack / email summary of the weekly retraining outcome.

    Reads champion and challenger metrics from XCom and sends a
    formatted summary to the quant-research channel.
    """
    ti = context["ti"]
    challenger_metrics: dict = ti.xcom_pull(
        task_ids="evaluate_challenger", key=XCOM_CHALLENGER_METRICS
    ) or {}
    champion_metrics: dict = ti.xcom_pull(
        task_ids="compare_champion_challenger", key=XCOM_CHAMPION_METRICS
    ) or {}
    execution_date: str = str(context.get("ds", datetime.utcnow().date()))

    promoted = context["dag_run"].conf.get("promoted", False)

    msg = (
        f"*Weekly Retraining Summary ({execution_date})*\n"
        f"Champion RMSE : {champion_metrics.get('surface_rmse', 'N/A')}\n"
        f"Challenger RMSE: {challenger_metrics.get('surface_rmse', 'N/A')}\n"
        f"Challenger directional accuracy: {challenger_metrics.get('directional_accuracy', 'N/A')}\n"
        f"Action: {'Promoted to champion' if promoted else 'Champion retained — challenger archived'}\n"
    )
    logger.info("Sending notification:\n%s", msg)

    # --- placeholder ---
    # slack_client.post_message(channel="#quant-research", text=msg)
    # email_client.send(to=["narehavag@gmail.com"], subject="Weekly Retraining Summary", body=msg)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="weekly_model_retraining",
    description=(
        "Weekly walk-forward model retraining with champion-challenger "
        "promotion logic."
    ),
    schedule_interval="0 2 * * 0",  # Sunday 2 AM UTC
    start_date=datetime(2024, 1, 7),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["volatility", "training", "weekly"],
) as dag:

    validate_training_data = PythonOperator(
        task_id="validate_training_data",
        python_callable=_validate_training_data,
        doc_md="Validate historical dataset completeness and quality.",
    )

    run_walk_forward_training = PythonOperator(
        task_id="run_walk_forward_training",
        python_callable=_run_walk_forward_training,
        doc_md="Execute walk-forward training and save challenger checkpoint.",
    )

    evaluate_challenger = PythonOperator(
        task_id="evaluate_challenger",
        python_callable=_evaluate_challenger,
        doc_md="Evaluate challenger model on held-out test set.",
    )

    compare_champion_challenger = BranchPythonOperator(
        task_id="compare_champion_challenger",
        python_callable=_compare_champion_challenger,
        doc_md="Compare metrics and branch to promote or archive.",
    )

    promote_if_better = PythonOperator(
        task_id="promote_if_better",
        python_callable=_promote_if_better,
        doc_md="Promote challenger to champion in the model registry.",
    )

    archive_challenger = PythonOperator(
        task_id="archive_challenger",
        python_callable=_archive_challenger,
        doc_md="Archive challenger model without promoting.",
    )

    notify_team = PythonOperator(
        task_id="notify_team",
        python_callable=_notify_team,
        trigger_rule="none_failed_min_one_success",  # runs after either branch
        doc_md="Notify team of retraining outcome via Slack / email.",
    )

    # ---------------------------------------------------------------------------
    # Dependency graph
    # ---------------------------------------------------------------------------
    (
        validate_training_data
        >> run_walk_forward_training
        >> evaluate_challenger
        >> compare_champion_challenger
        >> [promote_if_better, archive_challenger]
    )
    promote_if_better >> notify_team
    archive_challenger >> notify_team
