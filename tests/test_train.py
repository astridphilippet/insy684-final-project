"""Training runs end to end on the sample data and logs to MLflow."""
import mlflow
import pandas as pd

from src.data import preprocess
from src.train import train


def test_train_logs_run(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    df = preprocess(pd.read_csv("tests/data/sample_crimes.csv"))
    run_id = train(df, {"n_estimators": 10}, register=False)
    metrics = mlflow.get_run(run_id).data.metrics
    assert {"rmse", "mae", "baseline_rmse_mean"} <= set(metrics)
