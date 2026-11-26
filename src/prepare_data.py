"""Turn the raw SPVM crime file (French columns) into data/crimes.csv.

Works with the city's CSV or an Excel copy of it:
    python -m src.prepare_data path/to/actes-criminels.xlsx
"""
import sys

import pandas as pd

COLUMNS = {
    "CATEGORIE": "category",
    "DATE": "date",
    "QUART": "shift",
    "PDQ": "police_district",
    "X": "x_mtm8",
    "Y": "y_mtm8",
    "LONGITUDE": "longitude",
    "LATITUDE": "latitude",
}
CATEGORIES = {
    "Introduction": "Break and enter",
    "Vol dans / sur véhicule à moteur": "Theft in / from a motor vehicle",
    "Vol de véhicule à moteur": "Motor vehicle theft",
    "Méfait": "Mischief",
    "Vols qualifiés": "Confirmed theft",
    "Infractions entrainant la mort": "Offences causing death",
}
SHIFTS = {"jour": "day", "soir": "evening", "nuit": "night"}


def read_raw(path: str) -> pd.DataFrame:
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def fix_accents(text):
    """Repair accents garbled by opening the file on a Mac ('M√©fait' -> 'Méfait')."""
    try:
        return text.encode("mac_roman").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError, AttributeError):
        return text


def translate(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=COLUMNS)[list(COLUMNS.values())].copy()
    df["category"] = df["category"].map(fix_accents)
    # Unknown values keep their French name instead of becoming empty.
    df["category"] = df["category"].map(CATEGORIES).fillna(df["category"])
    df["shift"] = df["shift"].map(SHIFTS).fillna(df["shift"])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.dropna(subset=["date", "category"])


if __name__ == "__main__":
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "data/crimes.csv"
    df = translate(read_raw(src))
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out}")
