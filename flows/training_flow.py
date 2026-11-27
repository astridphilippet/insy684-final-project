"""Full training pipeline.

Steps: 
    1. Load and validate the raw crimes data 
    2. Build daily features 
    3. Train one model per parameter set 
    4. Mark the best registered version as 'champion'.

Run it from the repo root:
    python -m flows.training_flow --data data/crimes_montreal_en.csv
"""
import argparse

from prefect import flow, task

from src.data import load_crimes, preprocess
from src.train import promote_best, train

DEFAULT_GRID = [
    {"n_estimators": n, "max_depth": d}
    for n in (100, 300)
    for d in (5, 10, None)
]


@task
def load_data(path: str):
    return load_crimes(path)


@task
def build_features(raw):
    return preprocess(raw)


@task
def train_model(df, params: dict, register: bool = True) -> str:
    return train(df, params, register=register)


@task
def promote_champion():
    promote_best()


@flow(name="training-flow")
def training_flow(
    data_path: str = "data/crimes_montreal_en.csv",
    param_grid: list[dict] | None = None,
    register: bool = True,
) -> list[str]:
    """Run the pipeline and return the MLflow run ids."""
    df = build_features(load_data(data_path))
    run_ids = [train_model(df, p, register) for p in (param_grid or DEFAULT_GRID)]
    if register:
        promote_champion()
    return run_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/crimes_montreal_en.csv")
    args = parser.parse_args()
    training_flow(data_path=args.data)