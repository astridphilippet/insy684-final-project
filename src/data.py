"""Load, validate and preprocess the Montreal crime data.

Raw data has one row per reported crime. The model predicts how many crimes
are reported per day, so preprocessing turns the raw rows into one row per
day with a count and a few calendar and history features.
"""
import pandas as pd

REQUIRED_COLUMNS = [
    "category", "date", "shift", "police_district",
    "x_mtm8", "y_mtm8", "longitude", "latitude",
]
TARGET = "crime_count"
FEATURES = ["day_of_week", "month", "is_weekend", "lag_1", "lag_7", "rolling_mean_7"]


def validate_crimes(df: pd.DataFrame) -> None:
    """Raise ValueError listing every problem found, so bad data stops early."""
    problems = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    if df.empty:
        problems.append("file has no rows")
    for col in ("date", "category"):
        if col in df.columns and df[col].isna().any():
            problems.append(f"{int(df[col].isna().sum())} missing values in '{col}'")
    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], format="mixed", errors="coerce")
        bad = parsed.isna() & df["date"].notna()
        if bad.any():
            problems.append(f"{int(bad.sum())} values in 'date' are not valid dates")
    if problems:
        raise ValueError("Invalid crime data: " + "; ".join(problems))


def load_crimes(path: str) -> pd.DataFrame:
    """Read the crime CSV and check it before anything else uses it."""
    df = pd.read_csv(path)
    validate_crimes(df)
    return df


def daily_counts(raw: pd.DataFrame) -> pd.Series:
    """Number of crimes per calendar day. Days with no crimes count as 0."""
    dates = pd.to_datetime(raw["date"]).dt.normalize()
    counts = dates.value_counts().sort_index()
    full_range = pd.date_range(counts.index.min(), counts.index.max(), freq="D")
    counts = counts.reindex(full_range, fill_value=0)
    counts.name = TARGET
    return counts


def add_features(counts: pd.Series) -> pd.DataFrame:
    """Calendar and history features for each day.

    History features use only earlier days, so the model never sees the
    count it is trying to predict. The first 7 days have no full history
    and are dropped.
    """
    df = counts.to_frame(TARGET)
    df["day_of_week"] = df.index.day_of_week
    df["month"] = df.index.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["lag_1"] = df[TARGET].shift(1)
    df["lag_7"] = df[TARGET].shift(7)
    df["rolling_mean_7"] = df[TARGET].shift(1).rolling(7).mean()
    return df.dropna()


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw crime rows -> one row per day, ready for training."""
    return add_features(daily_counts(raw))
