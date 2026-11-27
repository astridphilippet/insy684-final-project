import mlflow
import pytest
from prefect.testing.utilities import prefect_test_harness

from flows.training_flow import training_flow

SAMPLE_DATA = "tests/data/sample_crimes.csv"

# Throwaway Prefect backend so tests never touch a real server.
@pytest.fixture(scope="module", autouse=True)
def prefect_local():
    with prefect_test_harness():
        yield


@pytest.fixture
def run_metrics(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    run_ids = training_flow(
        data_path=SAMPLE_DATA,
        param_grid=[{"n_estimators": 50, "max_depth": 5}],
        register=False,
    )
    assert len(run_ids) == 1
    return mlflow.get_run(run_ids[0]).data.metrics


def test_flow_logs_run(run_metrics):
    assert {"rmse", "mae", "baseline_rmse_mean"} <= set(run_metrics)

# Naive baseline predicts the training-set average every day.
def test_model_beats_naive_baseline(run_metrics):
    assert run_metrics["rmse"] < run_metrics["baseline_rmse_mean"]