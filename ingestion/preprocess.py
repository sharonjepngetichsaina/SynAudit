"""Turn the raw UCI downloads into the canonical CSVs the producer streams.

Reads  data/raw/{adult,credit-default}.csv
Writes data/processed/{adult,credit-default}.csv

Canonical form, as defined in docs/stream-contract.md:
  - column names lowercase with underscores
  - missing values as empty cells, serialised to JSON null by the producer
  - no encoding or scaling (that is each consumer's own business)
  - target column always present

Run from the repository root:  python ingestion/preprocess.py
"""

from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

# Credit Default ships with placeholder headers X1..X23. Real names come from
# the dataset documentation. Note there is no pay_1: the original data skips it.
CREDIT_COLUMNS = {
    "X1": "limit_bal",
    "X2": "sex",
    "X3": "education",
    "X4": "marriage",
    "X5": "age",
    "X6": "pay_0",
    "X7": "pay_2",
    "X8": "pay_3",
    "X9": "pay_4",
    "X10": "pay_5",
    "X11": "pay_6",
    "X12": "bill_amt1",
    "X13": "bill_amt2",
    "X14": "bill_amt3",
    "X15": "bill_amt4",
    "X16": "bill_amt5",
    "X17": "bill_amt6",
    "X18": "pay_amt1",
    "X19": "pay_amt2",
    "X20": "pay_amt3",
    "X21": "pay_amt4",
    "X22": "pay_amt5",
    "X23": "pay_amt6",
}

# A census sampling weight: how many people the row represents. Not a feature.
ADULT_DROP = ["fnlwgt"]


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in df.columns]
    return df


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and turn '?' into a real missing value."""
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"?": pd.NA, "": pd.NA})
    return df


def prepare_adult() -> pd.DataFrame:
    df = pd.read_csv(RAW / "adult.csv")
    df = normalise_columns(df)
    df = df.drop(columns=[c for c in ADULT_DROP if c in df.columns])
    df = clean_strings(df)

    # ucimlrepo returns train and test rows combined. Test rows carry a
    # trailing full stop ('<=50K.'), which would give four classes, not two.
    if "income" in df.columns:
        df["income"] = df["income"].str.rstrip(".")

    return df


def prepare_credit() -> pd.DataFrame:
    df = pd.read_csv(RAW / "credit-default.csv")
    df = df.rename(columns=CREDIT_COLUMNS)
    df = normalise_columns(df)
    df = clean_strings(df)
    return df


def report(name: str, df: pd.DataFrame, target: str) -> None:
    print(f"\n=== {name} ===")
    print(f"rows: {df.shape[0]:,}   columns: {df.shape[1]}")

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("missing values: none")
    else:
        print("missing values:")
        for col, n in missing.items():
            print(f"  {col}: {n:,} ({n / len(df):.1%})")

    duplicates = df.duplicated().sum()
    print(f"exact duplicate rows: {duplicates:,}")

    print(f"target '{target}':")
    for value, n in df[target].value_counts(dropna=False).items():
        print(f"  {value}: {n:,} ({n / len(df):.1%})")


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    datasets = {
        "adult": (prepare_adult(), "income"),
        "credit-default": (prepare_credit(), "default_payment_next_month"),
    }

    for name, (df, target) in datasets.items():
        if target not in df.columns:
            raise SystemExit(f"{name}: target column '{target}' missing after preprocessing")

        out = PROCESSED / f"{name}.csv"
        df.to_csv(out, index=False)
        report(name, df, target)
        print(f"written: {out}")
        print(f"columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
