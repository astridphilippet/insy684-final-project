"""Preprocessing checked against small inputs whose correct output is known."""
import pandas as pd

from src.data import FEATURES, TARGET, add_features, daily_counts, preprocess


def test_daily_counts_known_input():
    # Two crimes on Jan 1, none on Jan 2, one on Jan 3.
    raw = pd.DataFrame({"date": ["2024-01-01", "2024-01-01", "2024-01-03"]})
    counts = daily_counts(raw)
    expected = pd.Series(
        [2, 0, 1],
        index=pd.date_range("2024-01-01", "2024-01-03", freq="D"),
        name=TARGET,
    )
    pd.testing.assert_series_equal(counts, expected, check_freq=False)


def test_add_features_known_input():
    # Ten days with counts 1, 2, ..., 10 starting Monday 2024-01-01.
    days = pd.date_range("2024-01-01", periods=10, freq="D")
    counts = pd.Series(range(1, 11), index=days, name=TARGET)
    df = add_features(counts)

    # The first 7 days lack a full week of history, so 3 rows remain.
    assert list(df.index) == list(days[7:])
    row = df.loc["2024-01-09"]  # a Tuesday, count 9
    assert row[TARGET] == 9
    assert row["day_of_week"] == 1
    assert row["is_weekend"] == 0
    assert row["lag_1"] == 8  # yesterday
    assert row["lag_7"] == 2  # same weekday last week
    assert row["rolling_mean_7"] == 5  # mean of days 2..8


def test_preprocess_sample_file_has_no_gaps():
    raw = pd.read_csv("tests/data/sample_crimes.csv")
    df = preprocess(raw)
    assert set(FEATURES) <= set(df.columns)
    assert df.isna().sum().sum() == 0
    # Total crimes in the kept days must match the raw rows for those days.
    kept = pd.to_datetime(raw["date"]).isin(df.index)
    assert df[TARGET].sum() == kept.sum()
