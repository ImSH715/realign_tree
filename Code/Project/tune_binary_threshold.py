import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_csv", required=True)
    p.add_argument("--positive_label", default="1")
    p.add_argument("--output_csv", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.pred_csv)

    prob_col = f"prob_{args.positive_label}"
    if prob_col not in df.columns:
        raise ValueError(f"Missing column: {prob_col}. Available: {df.columns.tolist()}")

    y_true = df["y_true"].astype(int).values
    prob = df[prob_col].astype(float).values

    prevalence = float((y_true == 1).mean()) if len(y_true) else 0.0
    all_positive_f1 = (
        2 * prevalence / (1 + prevalence)
        if prevalence > 0
        else 0.0
    )

    rows = []
    for th in np.linspace(0.05, 0.95, 91):
        y_pred = (prob >= th).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=1, zero_division=0
        )
        acc = accuracy_score(y_true, y_pred)

        rows.append({
            "threshold": th,
            "accuracy": acc,
            "precision_shihuahuaco": precision,
            "recall_shihuahuaco": recall,
            "f1_shihuahuaco": f1,
            "pred_positive": int(y_pred.sum()),
            "positive_prevalence": prevalence,
            "f1_all_positive_baseline": all_positive_f1,
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.output_csv, index=False)

    best = out.sort_values("f1_shihuahuaco", ascending=False).iloc[0]
    print(f"Positive prevalence: {prevalence:.6f}")
    print(f"All-positive F1 baseline: {all_positive_f1:.6f}")
    if len(np.unique(y_true)) == 2:
        print(f"Average precision (PR-AUC): {average_precision_score(y_true, prob):.6f}")
        print(f"ROC-AUC: {roc_auc_score(y_true, prob):.6f}")
    print("Best threshold by Shihuahuaco F1:")
    print(best)


if __name__ == "__main__":
    main()
