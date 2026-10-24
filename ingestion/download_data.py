"""Download the SynAudit source datasets from the UCI repository into data/raw/.

Raw download only — no cleaning. Preprocessing to the canonical CSV is a separate step.
Run from the repository root:  python ingestion/download_data.py
"""

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

RAW = Path("data/raw")

# name -> (UCI dataset id, target column name to use in the saved CSV)
DATASETS = {
    "adult": (2, "income"),
    "credit-default": (350, "default_payment_next_month"),
}


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    for name, (uci_id, target) in DATASETS.items():
        print(f"fetching {name} (id={uci_id}) ...")
        ds = fetch_ucirepo(id=uci_id)

        df = ds.data.features.copy()
        df[target] = ds.data.targets.iloc[:, 0].values

        out = RAW / f"{name}.csv"
        df.to_csv(out, index=False)
        print(f"  {df.shape[0]:,} rows x {df.shape[1]} cols -> {out}")
        print(f"  columns: {list(df.columns)}\n")


if __name__ == "__main__":
    main()
