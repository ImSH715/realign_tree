"""
Export raw crop datasets from MIL realignment CSV outputs.

This is complementary to make_realign_debug_patches.py. Contact sheets are good
for visual QA, while this script creates clean individual patch images plus a
manifest that can be manually annotated and used for downstream classifiers.

Typical outputs:
- selected patches per model: output_dir/selected/<model>/*.png
- original point patches once per source point: output_dir/original/*.png
- patch_manifest.csv with probabilities, offsets, coordinates, and blank
  manual-label columns.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from tqdm import tqdm


def safe_mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_name(value, max_len=120):
    text = str(value).strip()
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return (out[:max_len] or "value").strip("_") or "value"


def row_float(row, key, default=0.0):
    value = row.get(key, default)
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def read_patch_from_src(src, px, py, patch_size):
    half = patch_size // 2
    col0 = int(round(px)) - half
    row0 = int(round(py)) - half
    window = rasterio.windows.Window(col0, row0, patch_size, patch_size)
    arr = src.read(window=window, boundless=True, fill_value=0)
    if arr.shape[0] >= 3:
        arr = arr[:3]
    elif arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    else:
        raise ValueError(f"Invalid band count: {arr.shape}")
    arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = np.nanpercentile(arr, [1, 99])
        arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
        arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def source_key(row):
    for key in ["source_index", "idx", "tree_id", "id", "PCA"]:
        if key in row and not pd.isna(row[key]):
            return safe_name(row[key], max_len=60)
    folder = safe_name(row.get("Folder", row.get("folder", "")), max_len=30)
    filename = safe_name(row.get("File", row.get("file", "")), max_len=40)
    px = int(round(row_float(row, "center_px")))
    py = int(round(row_float(row, "center_py")))
    return f"{folder}_{filename}_{px}_{py}"


def patch_point(row, kind):
    center_px = row_float(row, "center_px")
    center_py = row_float(row, "center_py")
    if kind == "original":
        return center_px, center_py, 0.0, 0.0, 0.0
    if kind == "raw":
        dx = row_float(row, "raw_dx_px")
        dy = row_float(row, "raw_dy_px")
        return center_px + dx, center_py + dy, dx, dy, row_float(row, "raw_offset_m")
    if kind == "context":
        dx = row_float(row, "context_dx_px")
        dy = row_float(row, "context_dy_px")
        return center_px + dx, center_py + dy, dx, dy, row_float(row, "context_offset_m")
    if kind == "selected":
        dx = row_float(row, "selected_dx_px")
        dy = row_float(row, "selected_dy_px")
        return center_px + dx, center_py + dy, dx, dy, row_float(row, "selected_offset_m")
    raise ValueError(f"Unknown patch kind: {kind}")


def patch_prob(row, kind):
    if kind == "original":
        return row_float(row, "bag_prob_1")
    if kind == "raw":
        return row_float(row, "raw_selected_prob_1")
    if kind == "context":
        return row_float(row, "context_selected_prob_1")
    if kind == "selected":
        if str(row.get("selection", "raw")) == "context":
            return row_float(row, "context_selected_prob_1")
        return row_float(row, "raw_selected_prob_1")
    return row_float(row, "bag_prob_1")


def filter_rows(df, args):
    out = df.copy()
    if args.only_positive_threshold:
        out = out[pd.to_numeric(out["is_positive_at_threshold"], errors="coerce").fillna(0) == 1]
    if args.min_bag_prob is not None:
        out = out[pd.to_numeric(out["bag_prob_1"], errors="coerce").fillna(-1) >= args.min_bag_prob]
    if args.models:
        out = out[out["model_run"].astype(str).isin(args.models)]
    if args.limit_per_model > 0:
        if args.sort_by == "random":
            out = (
                out.groupby("model_run", group_keys=False)
                .apply(lambda g: g.sample(frac=1.0, random_state=args.seed).head(args.limit_per_model))
                .reset_index(drop=True)
            )
        else:
            sort_values = pd.to_numeric(out[args.sort_by], errors="coerce")
            out = out.assign(_sort_values=sort_values)
            out = (
                out.sort_values(["model_run", "_sort_values"], ascending=[True, args.ascending])
                .groupby("model_run", group_keys=False)
                .head(args.limit_per_model)
                .drop(columns=["_sort_values"])
            )
    return out.reset_index(drop=True)


def parse_args():
    p = argparse.ArgumentParser(description="Export raw crop patches from MIL realignment outputs.")
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--patch_kinds", nargs="+", default=["selected"], choices=["original", "raw", "context", "selected"])
    p.add_argument("--include_original_once", action="store_true", help="Export one original patch per source point.")
    p.add_argument("--only_positive_threshold", action="store_true")
    p.add_argument("--min_bag_prob", type=float, default=None)
    p.add_argument("--sort_by", default="bag_prob_1")
    p.add_argument("--ascending", action="store_true")
    p.add_argument("--limit_per_model", type=int, default=0, help="Rows per model. Use 0 for all.")
    p.add_argument("--patch_size_px", type=int, default=160)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    safe_mkdir(output_dir)

    df = pd.read_csv(input_csv)
    if "model_run" not in df.columns:
        df["model_run"] = input_csv.parent.name
    required = ["image_path", "center_px", "center_py", "model_run"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {missing}")

    df = filter_rows(df, args)
    if df.empty:
        raise ValueError("No rows selected.")

    manifest_rows = []
    seen_originals = set()
    src_cache = {}

    def get_src(path):
        if path not in src_cache:
            src_cache[path] = rasterio.open(path)
        return src_cache[path]

    try:
        iterator = tqdm(df.iterrows(), total=len(df), desc="Export patches", dynamic_ncols=True)
        for row_idx, (_, row) in enumerate(iterator):
            row = row.to_dict()
            model = str(row.get("model_run", "model"))
            source = source_key(row)
            image_path = str(row["image_path"])
            src = get_src(image_path)

            kinds = list(args.patch_kinds)
            if args.include_original_once:
                original_key = (source, image_path, int(round(row_float(row, "center_px"))), int(round(row_float(row, "center_py"))))
                if original_key not in seen_originals:
                    seen_originals.add(original_key)
                    kinds = ["original"] + kinds

            for kind in kinds:
                if kind == "original" and not args.include_original_once and "original" not in args.patch_kinds:
                    continue
                px, py, dx_px, dy_px, offset_m = patch_point(row, kind)
                if kind == "original" and args.include_original_once:
                    subdir = output_dir / "original"
                    filename = f"{source}.png"
                else:
                    subdir = output_dir / kind / safe_name(model)
                    filename = f"{row_idx:06d}_{source}.png"
                safe_mkdir(subdir)
                out_path = subdir / filename
                if not (args.skip_existing and out_path.exists()):
                    img = read_patch_from_src(src, px, py, args.patch_size_px)
                    img.save(out_path)

                manifest_rows.append(
                    {
                        "patch_path": str(out_path),
                        "patch_kind": kind,
                        "source_key": source,
                        "model_run": "" if kind == "original" and args.include_original_once else model,
                        "image_path": image_path,
                        "patch_px": px,
                        "patch_py": py,
                        "dx_px": dx_px,
                        "dy_px": dy_px,
                        "offset_m": offset_m,
                        "bag_prob_1": row_float(row, "bag_prob_1"),
                        "patch_prob_1": patch_prob(row, kind),
                        "is_positive_at_threshold": int(row_float(row, "is_positive_at_threshold")),
                        "selection": row.get("selection", ""),
                        "manual_label": "",
                        "is_shihuahuaco": "",
                        "is_broadleaf_sink": "",
                        "notes": "",
                    }
                )
    finally:
        for src in src_cache.values():
            src.close()

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "patch_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Rows selected        : {len(df)}")
    print(f"Patches exported     : {len(manifest)}")
    print(f"Manifest             : {manifest_path}")


if __name__ == "__main__":
    main()
