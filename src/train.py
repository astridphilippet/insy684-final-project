"""Train the daily crime-count model and log each run to MLflow."""
import argparse
import itertools

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data import FEATURES, TARGET, load_crimes, preprocess

MODEL_NAME = "insy684-model"
EXPERIMENT = "crime-count"


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def time_split(df, test_frac=0.2):
    """Oldest days train, most recent days test (no shuffling: lag features)."""
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]


def train(df, params, register=True):
    """Fit one model, log it to MLflow, return the run id."""
    train_df, test_df = time_split(df)
    # Floats for every feature, so the API accepts whole numbers or decimals.
    X_tr, y_tr = train_df[FEATURES].astype("float64"), train_df[TARGET]
    X_te, y_te = test_df[FEATURES].astype("float64"), test_df[TARGET]

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        model = RandomForestRegressor(random_state=42, **params).fit(X_tr, y_tr)
        preds = model.predict(X_te)

        mlflow.log_params(params)
        mlflow.log_metrics({
            "rmse": rmse(y_te, preds),
            "mae": mean_absolute_error(y_te, preds),
            "baseline_rmse_mean": rmse(y_te, [y_tr.mean()] * len(y_te)),
            "baseline_rmse_lag7": rmse(y_te, X_te["lag_7"]),
        })
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(X_tr, preds),
            input_example=X_tr.head(3),
            registered_model_name=MODEL_NAME if register else None,
            # MLflow 3 saves sklearn models with skops and only allows
            # listed types; this is our own model, so trust its tree nodes.
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        return run.info.run_id


def promote_best():
    """Point the 'champion' alias at the registered version with the lowest RMSE."""
    best = mlflow.search_runs(
        experiment_names=[EXPERIMENT],
        # Newest run wins a tie, so retraining replaces an equal older model.
        order_by=["metrics.rmse ASC", "attributes.start_time DESC"],
    ).iloc[0]
    client = MlflowClient()
    for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
        if v.run_id == best.run_id:
            client.set_registered_model_alias(MODEL_NAME, "champion", v.version)
            print(f"champion = version {v.version} (rmse {best['metrics.rmse']:.2f})")
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/crimes.csv")
    args = parser.parse_args()

    df = preprocess(load_crimes(args.data))
    for n, depth in itertools.product([100, 300], [5, 10, None]):
        train(df, {"n_estimators": n, "max_depth": depth})
    promote_best()
