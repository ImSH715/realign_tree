import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_csv", required=True)
parser.add_argument("--output_csv", required=True)
parser.add_argument("--target_class", default="Shihuahuaco")
args = parser.parse_args()

df = pd.read_csv(args.input_csv)

df["BinaryTree"] = (df["Tree"] == args.target_class).astype(int)

df.to_csv(args.output_csv, index=False)

print("Saved:", args.output_csv)
print(df["BinaryTree"].value_counts())