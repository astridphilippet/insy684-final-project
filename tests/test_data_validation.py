import pandas as pd
import pytest

DATA_PATH = "data/crimes_montreal_en.csv"
REQUIRED_COLUMNS = [
    "category", "date", "shift", "police_district",
    "x_mtm8", "y_mtm8", "longitude", "latitude",
]
TARGET = "date"  # the daily crime count is built from this column


@pytest.fixture
def df():
    return pd.read_csv(DATA_PATH)


def test_required_columns_present(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"


def test_target_has_no_missing_values(df):
    assert df[TARGET].notna().all()
