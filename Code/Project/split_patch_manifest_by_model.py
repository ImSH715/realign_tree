"""
Split an exported patch manifest into one CSV per model with image links.

The generated CSVs are intended for manual labelling in Excel/LibreOffice.
Each selected-patch row includes:
- selected_image_link: hyperlink to the model-selected patch
- original_image_link: hyperlink to the matching original-coordinate patch

Run this after export_realign_patch_dataset.py.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


DATASET_MARKER = "patch_dataset_all500_per_model/"


def safe_name(value):
    text = str(value).strip()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:160] or "model"


def rel_inside_dataset(path):
    text = str(path)
    text = text.replace("\\", "/")
    idx = text.find(DATASET_MARKER)
    if idx >= 0:
        text = text[idx + len(DATASET_MARKER) :]
    return text.replace("/", "\\")


def hyperlink(rel_from_output, label):
    escaped_path = str(rel_from_output).replace('"', '""')
    escaped_label = str(label).replace('"', '""')
    return f'=HYPERLINK("{escaped_path}","{escaped_label}")'


def parse_args():
    p = argparse.ArgumentParser(description="Split patch manifest by model and add image hyperlink columns.")
    p.add_argument("--manifest", default="patch_dataset_all500_per_model/patch_manifest.csv")
    p.add_argument("--output_dir", default="patch_dataset_all500_per_model/manifests_by_model")
    p.add_argument("--dataset_name", default="patch_dataset_all500_per_model")
    return p.parse_args()


def main():
    args = parse_args()
    manifest = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest)
    if "source_uid" not in df.columns:
        df["source_uid"] = df["source_key"]

    df["local_relative_path"] = df["patch_path"].map(rel_inside_dataset)
    df["image_link"] = df["local_relative_path"].map(lambda p: hyperlink(Path("..") / p, "open"))

    original = df[df["patch_kind"] == "original"].copy()
    original = original.sort_values("source_uid")
    original.to_csv(output_dir / "original_patches.csv", index=False)

    original_by_uid = dict(zip(original["source_uid"].astype(str), original["local_relative_path"]))
    selected = df[df["patch_kind"] == "selected"].copy()

    index_rows = []
    for model, group in selected.groupby("model_run", sort=True):
        out = group.sort_values("source_uid").copy()
        out["selected_image_link"] = out["local_relative_path"].map(lambda p: hyperlink(Path("..") / p, "selected"))
        out["original_local_relative_path"] = out["source_uid"].astype(str).map(original_by_uid).fillna("")
        out["original_image_link"] = out["original_local_relative_path"].map(
            lambda p: hyperlink(Path("..") / p, "original") if p else ""
        )

        filename = f"{safe_name(model)}.csv"
        out.to_csv(output_dir / filename, index=False)
        index_rows.append(
            {
                "model_run": model,
                "rows": len(out),
                "csv_file": str(Path(output_dir.name) / filename),
            }
        )

    pd.DataFrame(index_rows).to_csv(output_dir / "model_index.csv", index=False)
    print(f"Wrote {len(index_rows)} model CSVs to {output_dir}")
    print(f"Original rows: {len(original)}")
    print(f"Selected rows: {len(selected)}")


if __name__ == "__main__":
    main()
