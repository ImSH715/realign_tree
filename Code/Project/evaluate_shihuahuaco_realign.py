import argparse
import pandas as pd
import numpy as np


def dist_m(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--label_column", default="label")
    p.add_argument("--target_label", default="Shihuahuaco")
    p.add_argument("--gt_x", default="gt_east")
    p.add_argument("--gt_y", default="gt_north")
    p.add_argument("--original_x", default="original_east")
    p.add_argument("--original_y", default="original_north")
    p.add_argument("--refined_x", default="refined_east")
    p.add_argument("--refined_y", default="refined_north")
    p.add_argument("--output_csv", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.input_csv)

    df[args.label_column] = df[args.label_column].astype(str).str.strip()
    sub = df[df[args.label_column] == args.target_label].copy()

    if len(sub) == 0:
        raise ValueError(f"No rows found for target label: {args.target_label}")

    sub["dist_before_m"] = dist_m(
        sub[args.original_x], sub[args.original_y],
        sub[args.gt_x], sub[args.gt_y],
    )

    sub["dist_after_m"] = dist_m(
        sub[args.refined_x], sub[args.refined_y],
        sub[args.gt_x], sub[args.gt_y],
    )

    sub["improvement_m"] = sub["dist_before_m"] - sub["dist_after_m"]
    sub["improved"] = sub["improvement_m"] > 0

    for tol in [1, 2, 5, 10, 20]:
        sub[f"success_within_{tol}m_before"] = sub["dist_before_m"] <= tol
        sub[f"success_within_{tol}m_after"] = sub["dist_after_m"] <= tol

    sub.to_csv(args.output_csv, index=False)

    print("=" * 80)
    print(f"Target label: {args.target_label}")
    print(f"N: {len(sub)}")
    print("-" * 80)
    print(f"Mean before:  {sub['dist_before_m'].mean():.3f} m")
    print(f"Mean after:   {sub['dist_after_m'].mean():.3f} m")
    print(f"Median before:{sub['dist_before_m'].median():.3f} m")
    print(f"Median after: {sub['dist_after_m'].median():.3f} m")
    print(f"Mean improvement: {sub['improvement_m'].mean():.3f} m")
    print(f"Improved rate: {sub['improved'].mean() * 100:.2f}%")
    print("-" * 80)

    for tol in [1, 2, 5, 10, 20]:
        before = sub[f"success_within_{tol}m_before"].mean() * 100
        after = sub[f"success_within_{tol}m_after"].mean() * 100
        print(f"Within {tol:02d}m | before: {before:6.2f}% | after: {after:6.2f}%")

    print("=" * 80)
    print("Saved:", args.output_csv)


if __name__ == "__main__":
    main()