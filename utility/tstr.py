"""ML utility check.

Trains a Hoeffding tree on synthetic data and tests it on real data (TSTR),
and compares it with a tree trained on real data (TRTR).
Hoeffding tree: Domingos and Hulten, 2000, "Mining High-Speed Data Streams".
"""

import argparse
import random
from pathlib import Path

import pandas as pd
from river import metrics, tree

TARGETS = {
    "adult": "income",
    "credit-default": "default_payment_next_month",
}


def load_data(dataset):
    """Read the real and synthetic CSV files for one dataset."""
    real = pd.read_csv(Path("data/processed") / f"{dataset}.csv")
    synthetic = pd.read_csv(Path("data/synthetic") / f"{dataset}-synthetic.csv")
    return real, synthetic


def to_label(value, dataset):
    """Turn the label into True (the rare class) or False."""
    if dataset == "adult":
        return str(value).strip() == ">50K"
    return int(value) == 1


def to_rows(df, dataset):
    """Turn a table into a list of (features, label) pairs for river."""
    target = TARGETS[dataset]
    rows = []
    for record in df.to_dict("records"):
        y = to_label(record.pop(target), dataset)
        x = {k: v for k, v in record.items() if pd.notna(v)}
        rows.append((x, y))
    return rows


def split_real(rows, test_share=0.2, seed=42):
    """Shuffle the real rows once and set aside a test set."""
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    n_test = int(len(rows) * test_share)
    return rows[n_test:], rows[:n_test]


def evaluate(model, test_rows):
    """Score one model on the held-out real rows."""
    acc = metrics.Accuracy()
    bal = metrics.BalancedAccuracy()
    f1 = metrics.F1()
    for x, y in test_rows:
        y_pred = model.predict_one(x)
        acc.update(y, y_pred)
        bal.update(y, y_pred)
        f1.update(y, y_pred)
    return acc.get(), bal.get(), f1.get()


def run(dataset, window=500, n_test=2000):
    """Stream both training sets and score both models every window."""
    real, synthetic = load_data(dataset)
    real_train, real_test = split_real(to_rows(real, dataset))
    synth_rows = to_rows(synthetic, dataset)
    test_rows = real_test[:n_test]

    model_s = tree.HoeffdingTreeClassifier()
    model_r = tree.HoeffdingTreeClassifier()
    results = []

    n = min(len(synth_rows), len(real_train))
    for i in range(n):
        x_s, y_s = synth_rows[i]
        x_r, y_r = real_train[i]
        model_s.learn_one(x_s, y_s)
        model_r.learn_one(x_r, y_r)
        if (i + 1) % window == 0:
            acc_s, bal_s, f1_s = evaluate(model_s, test_rows)
            acc_r, bal_r, f1_r = evaluate(model_r, test_rows)
            results.append({
                "dataset": dataset,
                "window": (i + 1) // window,
                "rows_seen": i + 1,
                "acc_synthetic": acc_s,
                "bal_acc_synthetic": bal_s,
                "f1_synthetic": f1_s,
                "acc_real": acc_r,
                "bal_acc_real": bal_r,
                "f1_real": f1_r,
                "utility_gap": bal_r - bal_s,
            })
            print(f"rows {i + 1}: S bal_acc {bal_s:.3f}  R bal_acc {bal_r:.3f}  gap {bal_r - bal_s:.3f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="TSTR utility check with Hoeffding trees.")
    parser.add_argument("--dataset", choices=list(TARGETS), default="adult")
    parser.add_argument("--window", type=int, default=500)
    args = parser.parse_args()

    results = run(args.dataset, window=args.window)

    out_dir = Path("results/utility")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}-utility.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved {len(results)} windows to {out_path}")


if __name__ == "__main__":
    main()