import argparse
import os
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--from_prefix", required=True)
    p.add_argument("--to_prefix", required=True)
    p.add_argument("--path_columns", nargs="+", default=["image_path", "matched_tif"])
    args = p.parse_args()

    df = pd.read_csv(args.input_csv).copy()

    found_any = False
    for col in args.path_columns:
        if col in df.columns:
            found_any = True
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(args.from_prefix, args.to_prefix, regex=False)
            )
            print(f"[INFO] Rewrote column: {col}")

    if not found_any:
        raise ValueError(
            f"None of the requested path columns exist. "
            f"Available columns: {df.columns.tolist()}"
        )

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"[INFO] Saved rewritten CSV: {args.output_csv}")
    print(df.head())


if __name__ == "__main__":
    main()