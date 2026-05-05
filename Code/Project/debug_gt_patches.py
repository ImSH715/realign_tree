import argparse
import os
import geopandas as gpd
import rasterio
import numpy as np
from PIL import Image
from tqdm import tqdm


def normalize_stem(name):
    return os.path.splitext(os.path.basename(str(name).strip()))[0].lower()


def build_tif_index(imagery_root):
    folder_to_paths = {}

    for root, _, files in os.walk(imagery_root):
        tif_files = [f for f in files if f.lower().endswith((".tif", ".tiff"))]
        if not tif_files:
            continue

        rel = os.path.relpath(root, imagery_root)
        parts = rel.split(os.sep)

        folder_key = None
        for p in parts:
            if p.startswith("2023-"):
                folder_key = p
                break

        if folder_key is None:
            folder_key = parts[0] if parts and parts[0] != "." else ""

        folder_to_paths.setdefault(folder_key, [])
        for f in tif_files:
            folder_to_paths[folder_key].append(os.path.join(root, f))

    return folder_to_paths


def resolve_tif(folder_to_paths, folder, filename):
    folder = str(folder).strip()
    stem = normalize_stem(filename)

    paths = folder_to_paths.get(folder, [])
    if not paths:
        raise FileNotFoundError(f"No TIFFs indexed for folder={folder}")

    exact, contains, reverse = [], [], []

    for p in paths:
        tif_stem = normalize_stem(p)
        if tif_stem == stem:
            exact.append(p)
        elif stem in tif_stem:
            contains.append(p)
        elif tif_stem in stem:
            reverse.append(p)

    if exact:
        return sorted(exact, key=len)[0]
    if contains:
        return sorted(contains, key=len)[0]
    if reverse:
        return sorted(reverse, key=len)[0]

    raise FileNotFoundError(f"No matching TIFF for folder={folder}, file={filename}")


def convert_to_pixel(src, x, y, coord_mode):
    x = float(x)
    y = float(y)

    if coord_mode == "pixel":
        return x, y, "pixel"

    if coord_mode == "normalized":
        return x * src.width, y * src.height, "normalized"

    if coord_mode == "world":
        row, col = src.index(x, y)
        return float(col), float(row), "world"

    if coord_mode == "auto":
        if 0 <= x <= 1 and 0 <= y <= 1:
            return x * src.width, y * src.height, "normalized"
        if 0 <= x < src.width and 0 <= y < src.height:
            return x, y, "pixel"
        row, col = src.index(x, y)
        return float(col), float(row), "world"

    raise ValueError(coord_mode)


def read_patch(path, x, y, patch_size, coord_mode):
    half = patch_size // 2

    with rasterio.open(path) as src:
        px, py, used_mode = convert_to_pixel(src, x, y, coord_mode)
        col0 = int(round(px)) - half
        row0 = int(round(py)) - half
        window = rasterio.windows.Window(col0, row0, patch_size, patch_size)
        arr = src.read(window=window, boundless=True, fill_value=0)

    if arr.shape[0] >= 3:
        arr = arr[:3]
    elif arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)

    arr = np.transpose(arr, (1, 2, 0))

    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = np.nanpercentile(arr, [1, 99])
        arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
        arr = (arr * 255).astype(np.uint8)

    return Image.fromarray(arr), px, py, used_mode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shp", required=True)
    p.add_argument("--imagery_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--label_field", default="Tree")
    p.add_argument("--folder_field", default="Folder")
    p.add_argument("--file_field", default="File")
    p.add_argument("--fx_field", default="fx")
    p.add_argument("--fy_field", default="fy")
    p.add_argument("--coord_mode", default="auto", choices=["auto", "normalized", "pixel", "world"])
    p.add_argument("--patch_size_px", type=int, default=224)
    p.add_argument("--max_patches", type=int, default=80)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    gdf = gpd.read_file(args.shp)
    gdf = gdf[gdf[args.label_field].notna()].copy()
    gdf[args.label_field] = gdf[args.label_field].astype(str).str.strip()

    folder_to_paths = build_tif_index(args.imagery_root)

    rows = []

    for i, (_, row) in enumerate(tqdm(gdf.iterrows(), total=min(len(gdf), args.max_patches))):
        if i >= args.max_patches:
            break

        label = str(row[args.label_field]).strip()
        safe_label = label.replace(" ", "_").replace("/", "_")

        try:
            image_path = resolve_tif(
                folder_to_paths,
                row[args.folder_field],
                row[args.file_field],
            )

            img, px, py, used_mode = read_patch(
                image_path,
                row[args.fx_field],
                row[args.fy_field],
                args.patch_size_px,
                args.coord_mode,
            )

            out_name = f"{i:04d}_{safe_label}.png"
            out_path = os.path.join(args.output_dir, out_name)
            img.save(out_path)

            rows.append({
                "idx": i,
                "label": label,
                "folder": row[args.folder_field],
                "file": row[args.file_field],
                "image_path": image_path,
                "raw_fx": row[args.fx_field],
                "raw_fy": row[args.fy_field],
                "pixel_x": px,
                "pixel_y": py,
                "coord_mode_used": used_mode,
                "patch": out_path,
            })

        except Exception as e:
            rows.append({
                "idx": i,
                "label": label,
                "error": str(e),
            })

    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "debug_patches.csv"), index=False)
    print(f"Saved debug patches to: {args.output_dir}")


if __name__ == "__main__":
    main()
