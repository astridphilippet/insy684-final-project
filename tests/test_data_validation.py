import os

import pandas as pd
import pytest

from src.data import REQUIRED_COLUMNS, load_crimes, validate_crimes

REAL_DATA = "data/crimes_montreal_en.csv"
SAMPLE_DATA = "tests/data/sample_crimes.csv"
# Use the real dataset when it's there (local runs); otherwise the small
# sample checked into the repo, so these tests run in CI instead of skipping.
DATA_PATH = REAL_DATA if os.path.exists(REAL_DATA) else SAMPLE_DATA


@pytest.fixture
def df():
    return pd.read_csv(DATA_PATH)


def test_required_columns_present(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"


def test_date_has_no_missing_values(df):
    # The daily crime count (the target) is built from this column.
    assert df["date"].notna().all()


def test_load_crimes_accepts_valid_file():
    assert len(load_crimes(DATA_PATH)) > 0


def test_validation_rejects_missing_column(df):
    with pytest.raises(ValueError, match="missing columns"):
        validate_crimes(df.drop(columns=["date"]))


def test_validation_rejects_missing_dates(df):
    bad = df.copy()
    bad.loc[0, "date"] = None
    with pytest.raises(ValueError, match="missing values in 'date'"):
        validate_crimes(bad)


def test_validation_rejects_unreadable_dates(df):
    bad = df.copy()
    bad.loc[0, "date"] = "not a date"
    with pytest.raises(ValueError, match="not valid dates"):
        validate_crimes(bad)
