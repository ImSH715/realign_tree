"""
Export paired patch folders for realigned coordinates and nearby weak species labels.

The MIL realignment output gives one selected coordinate per input point. This
script crops those selected coordinates and pairs them with crops at the nearest
weak census coordinates for one comparison species. When the comparison species
is not supplied, the script first finds the nearest non-target census point for
each realigned coordinate and uses the most common species among those matches.
"""

import argparse
import json
from collections import Counter
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


def read_csv_flexible(path):
    last_error = None
    for encoding in ["utf-8-sig", "utf-8", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def read_species_file(path):
    species = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            value = line.strip()
            if value and not value.startswith("#"):
                species.append(value)
    return species


def normalize_key(value):
    return str(value).strip().casefold()


def row_float(row, key, default=np.nan):
    try:
        value = row.get(key, default)
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


def dataframe_points(df, east_col, north_col):
    east = pd.to_numeric(df[east_col], errors="coerce")
    north = pd.to_numeric(df[north_col], errors="coerce")
    valid = east.notna() & north.notna()
    out = df.loc[valid].copy()
    out["_censo_row"] = out.index
    out = out.reset_index(drop=True)
    out["_east"] = east.loc[valid].astype(float).to_numpy()
    out["_north"] = north.loc[valid].astype(float).to_numpy()
    return out


def nearest_index(coords, x, y, candidate_mask=None):
    if candidate_mask is not None:
        indexes = np.flatnonzero(candidate_mask)
        if len(indexes) == 0:
            return None, np.nan
        compare = coords[indexes]
    else:
        indexes = None
        compare = coords
    d2 = (compare[:, 0] - x) ** 2 + (compare[:, 1] - y) ** 2
    local = int(np.argmin(d2))
    idx = local if indexes is None else int(indexes[local])
    return idx, float(np.sqrt(d2[local]))


def model_rows(df, models, model_contains):
    out = df.copy()
    if models:
        out = out[out["model_run"].astype(str).isin(models)]
    if model_contains:
        needles = [normalize_key(value) for value in model_contains]
        keep = out["model_run"].astype(str).map(
            lambda value: any(needle in normalize_key(value) for needle in needles)
        )
        out = out[keep]
    return out.reset_index(drop=True)


def pixel_from_world(src, x, y):
    row, col = src.index(float(x), float(y))
    return float(col), float(row)


def bounds_mask(coords, bounds):
    return (
        (coords[:, 0] >= bounds.left)
        & (coords[:, 0] <= bounds.right)
        & (coords[:, 1] >= bounds.bottom)
        & (coords[:, 1] <= bounds.top)
    )


def realigned_xy(row):
    x = row_float(row, "realigned_x")
    y = row_float(row, "realigned_y")
    if np.isfinite(x) and np.isfinite(y):
        return x, y
    x = row_float(row, "raw_realigned_x")
    y = row_float(row, "raw_realigned_y")
    if np.isfinite(x) and np.isfinite(y):
        return x, y
    raise ValueError("Realignment CSV needs realigned_x/realigned_y or raw_realigned_x/raw_realigned_y.")


def source_key(row, row_idx):
    for key in ["point_id", "source_uid", "source_key", "censo_source_index"]:
        value = row.get(key)
        if value is not None and not pd.isna(value):
            return safe_name(value, max_len=70)
    return f"row_{row_idx:06d}"


def export_patch(src, px, py, patch_size_px, out_path):
    safe_mkdir(out_path.parent)
    img = read_patch_from_src(src, px, py, patch_size_px)
    img.save(out_path)


def parse_args():
    p = argparse.ArgumentParser(
        description="Export realigned patches and nearest weak-species census patches into paired folders."
    )
    p.add_argument("--input_csv", required=True, help="CSV from apply_mil_realign.py or the combined all-model table.")
    p.add_argument("--censo_csv", required=True, help="Full forest census CSV with species and UTM coordinates.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--models", nargs="*", default=None, help="Exact model_run names to export.")
    p.add_argument(
        "--model_contains",
        nargs="*",
        default=None,
        help="Case-insensitive model_run substrings to export when exact names are inconvenient.",
    )
    p.add_argument("--weak_species", default="", help="Comparison species common name. Leave blank to infer the mode.")
    p.add_argument(
        "--candidate_species",
        nargs="*",
        default=None,
        help="Restrict the nearest-mode search to these census common names.",
    )
    p.add_argument(
        "--candidate_species_file",
        default="",
        help="UTF-8 text file with one candidate census common name per line.",
    )
    p.add_argument("--target_species", default="Shihuahuaco")
    p.add_argument("--censo_species_col", default="NOMBRE_COMUN")
    p.add_argument("--censo_scientific_col", default="NOMBRE_CIENTIFICO")
    p.add_argument("--censo_east_col", default="COORDENADA_ESTE")
    p.add_argument("--censo_north_col", default="COORDENADA_NORTE")
    p.add_argument("--censo_zone_col", default="ZONA_UTM")
    p.add_argument("--zone_value", default="18S")
    p.add_argument("--patch_size_px", type=int, default=160)
    p.add_argument(
        "--weak_same_tif_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require each weak comparison point to fall inside the realignment row's source TIFF.",
    )
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    safe_mkdir(output_dir)

    realign_df = pd.read_csv(args.input_csv)
    if "model_run" not in realign_df.columns:
        realign_df["model_run"] = Path(args.input_csv).parent.name
    required = ["model_run", "image_path"]
    missing = [col for col in required if col not in realign_df.columns]
    if missing:
        raise ValueError(f"Missing required realignment columns: {missing}")
    realign_df = model_rows(realign_df, args.models, args.model_contains)
    if realign_df.empty:
        available = sorted(pd.read_csv(args.input_csv)["model_run"].astype(str).unique())
        raise ValueError(f"No rows selected. Available model_run values include: {available[:20]}")

    censo = read_csv_flexible(args.censo_csv)
    required_censo = [args.censo_species_col, args.censo_east_col, args.censo_north_col]
    missing_censo = [col for col in required_censo if col not in censo.columns]
    if missing_censo:
        raise ValueError(f"Missing required census columns: {missing_censo}")
    if args.zone_value and args.censo_zone_col in censo.columns:
        censo = censo[censo[args.censo_zone_col].astype(str).str.upper() == args.zone_value.upper()].copy()
    censo = dataframe_points(censo, args.censo_east_col, args.censo_north_col)
    censo["_species_norm"] = censo[args.censo_species_col].map(normalize_key)
    target_norm = normalize_key(args.target_species)
    candidate_species = list(args.candidate_species or [])
    if args.candidate_species_file:
        candidate_species.extend(read_species_file(args.candidate_species_file))
    candidate_norms = {normalize_key(value) for value in candidate_species if str(value).strip()}
    comparison_mask = censo["_species_norm"].ne(target_norm) & censo[args.censo_species_col].notna()
    if candidate_norms:
        comparison_mask &= censo["_species_norm"].isin(candidate_norms)
    comparison_pool = censo[comparison_mask].copy().reset_index(drop=True)
    if comparison_pool.empty:
        raise ValueError("No candidate weak census species were available for nearest matching.")
    pool_coords = comparison_pool[["_east", "_north"]].to_numpy(dtype=float)

    nearest_any_rows = []
    nearest_counts = Counter()
    for row_idx, (_, row) in enumerate(
        tqdm(realign_df.iterrows(), total=len(realign_df), desc="Nearest census mode", dynamic_ncols=True)
    ):
        x, y = realigned_xy(row)
        nearest_idx, nearest_m = nearest_index(pool_coords, x, y)
        nearest = comparison_pool.iloc[int(nearest_idx)]
        species = str(nearest[args.censo_species_col]).strip()
        nearest_counts[species] += 1
        nearest_any_rows.append(
            {
                "row_idx": row_idx,
                "nearest_any_species": species,
                "nearest_any_censo_index": int(nearest_idx),
                "nearest_any_distance_m": nearest_m,
            }
        )
    nearest_any = pd.DataFrame(nearest_any_rows)

    if args.weak_species:
        weak_species = str(args.weak_species).strip()
    else:
        weak_species = nearest_counts.most_common(1)[0][0]
    weak_norm = normalize_key(weak_species)
    weak_pool = censo[censo["_species_norm"].eq(weak_norm)].copy().reset_index(drop=True)
    if weak_pool.empty:
        available = censo[args.censo_species_col].value_counts().head(25).index.tolist()
        raise ValueError(f"Weak species {weak_species!r} was not found. Top census species: {available}")
    weak_coords = weak_pool[["_east", "_north"]].to_numpy(dtype=float)

    manifest_rows = []
    pair_rows = []
    weak_rows_skipped = []
    src_cache = {}

    def get_src(path):
        if path not in src_cache:
            src_cache[path] = rasterio.open(path)
        return src_cache[path]

    try:
        iterator = tqdm(realign_df.iterrows(), total=len(realign_df), desc="Export paired patches", dynamic_ncols=True)
        for row_idx, (_, row) in enumerate(iterator):
            row_dict = row.to_dict()
            model_run = str(row_dict["model_run"])
            model_name = safe_name(model_run)
            pair_key = source_key(row_dict, row_idx)
            image_path = str(row_dict["image_path"])
            src = get_src(image_path)
            realigned_x, realigned_y = realigned_xy(row_dict)
            realigned_px, realigned_py = pixel_from_world(src, realigned_x, realigned_y)

            tif_mask = bounds_mask(weak_coords, src.bounds) if args.weak_same_tif_only else None
            weak_idx, weak_distance_m = nearest_index(weak_coords, realigned_x, realigned_y, tif_mask)
            nearest_any_row = nearest_any.iloc[row_idx].to_dict()

            realigned_path = (
                output_dir
                / "realigned"
                / model_name
                / f"{row_idx:06d}_{pair_key}.png"
            )
            if not (args.skip_existing and realigned_path.exists()):
                export_patch(src, realigned_px, realigned_py, args.patch_size_px, realigned_path)
            manifest_rows.append(
                {
                    "patch_path": str(realigned_path),
                    "patch_kind": "realigned",
                    "pair_key": pair_key,
                    "pair_row": row_idx,
                    "model_run": model_run,
                    "image_path": image_path,
                    "patch_x": realigned_x,
                    "patch_y": realigned_y,
                    "patch_px": realigned_px,
                    "patch_py": realigned_py,
                    "weak_species": weak_species,
                    **nearest_any_row,
                }
            )

            pair_row = {
                "pair_key": pair_key,
                "pair_row": row_idx,
                "model_run": model_run,
                "point_id": row_dict.get("point_id", ""),
                "realigned_patch_path": str(realigned_path),
                "realigned_x": realigned_x,
                "realigned_y": realigned_y,
                "weak_species": weak_species,
                **nearest_any_row,
            }
            if weak_idx is None:
                pair_row["weak_patch_path"] = ""
                pair_row["weak_species_distance_m"] = np.nan
                pair_row["weak_skip_reason"] = "no weak species coordinate inside source TIFF"
                pair_rows.append(pair_row)
                weak_rows_skipped.append(pair_row)
                continue

            weak = weak_pool.iloc[int(weak_idx)]
            weak_x = float(weak["_east"])
            weak_y = float(weak["_north"])
            weak_px, weak_py = pixel_from_world(src, weak_x, weak_y)
            weak_path = (
                output_dir
                / "weak_species"
                / safe_name(weak_species)
                / model_name
                / f"{row_idx:06d}_{pair_key}.png"
            )
            if not (args.skip_existing and weak_path.exists()):
                export_patch(src, weak_px, weak_py, args.patch_size_px, weak_path)

            weak_row = int(weak.get("_censo_row", weak.name))
            weak_source_index = weak.get("censo_source_index", weak_row)
            weak_scientific = weak.get(args.censo_scientific_col, "")
            pair_row.update(
                {
                    "weak_patch_path": str(weak_path),
                    "weak_x": weak_x,
                    "weak_y": weak_y,
                    "weak_px": weak_px,
                    "weak_py": weak_py,
                    "weak_species_distance_m": weak_distance_m,
                    "weak_censo_row": weak_row,
                    "weak_censo_source_index": weak_source_index,
                    "weak_scientific_name": weak_scientific,
                    "weak_skip_reason": "",
                }
            )
            pair_rows.append(pair_row)
            manifest_rows.append(
                {
                    "patch_path": str(weak_path),
                    "patch_kind": "weak_species",
                    "pair_key": pair_key,
                    "pair_row": row_idx,
                    "model_run": model_run,
                    "image_path": image_path,
                    "patch_x": weak_x,
                    "patch_y": weak_y,
                    "patch_px": weak_px,
                    "patch_py": weak_py,
                    "weak_species": weak_species,
                    "weak_species_distance_m": weak_distance_m,
                    "weak_censo_row": weak_row,
                    "weak_censo_source_index": weak_source_index,
                    "weak_scientific_name": weak_scientific,
                    **nearest_any_row,
                }
            )
    finally:
        for src in src_cache.values():
            src.close()

    manifest = pd.DataFrame(manifest_rows)
    pairs = pd.DataFrame(pair_rows)
    manifest_path = output_dir / "patch_manifest.csv"
    pair_path = output_dir / "patch_pairs.csv"
    nearest_counts_path = output_dir / "nearest_non_target_species_counts.csv"
    manifest.to_csv(manifest_path, index=False)
    pairs.to_csv(pair_path, index=False)
    (
        pd.DataFrame(nearest_counts.most_common(), columns=["species", "count"])
        .to_csv(nearest_counts_path, index=False)
    )

    model_counts = realign_df["model_run"].astype(str).value_counts().sort_index().to_dict()
    summary = {
        "input_csv": str(args.input_csv),
        "censo_csv": str(args.censo_csv),
        "output_dir": str(output_dir),
        "models": {str(k): int(v) for k, v in model_counts.items()},
        "realigned_rows": int(len(realign_df)),
        "target_species": args.target_species,
        "candidate_species": sorted(candidate_species),
        "candidate_species_file": args.candidate_species_file,
        "candidate_species_censo_rows": int(len(comparison_pool)),
        "auto_weak_species": not bool(args.weak_species),
        "weak_species": weak_species,
        "weak_species_censo_rows": int(len(weak_pool)),
        "nearest_non_target_species_top": [
            {"species": str(species), "count": int(count)}
            for species, count in nearest_counts.most_common(20)
        ],
        "realigned_patches": int((manifest["patch_kind"] == "realigned").sum()),
        "weak_species_patches": int((manifest["patch_kind"] == "weak_species").sum()),
        "weak_species_rows_skipped": int(len(weak_rows_skipped)),
        "weak_same_tif_only": bool(args.weak_same_tif_only),
        "patch_size_px": int(args.patch_size_px),
        "manifest": str(manifest_path),
        "pairs": str(pair_path),
        "nearest_non_target_species_counts": str(nearest_counts_path),
    }
    summary_path = output_dir / "patch_pair_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Realigned patches : {output_dir / 'realigned'}")
    print(f"Weak species       : {output_dir / 'weak_species'}")
    print(f"Manifest           : {manifest_path}")
    print(f"Pairs              : {pair_path}")
    print(f"Summary            : {summary_path}")


if __name__ == "__main__":
    main()
