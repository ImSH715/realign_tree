"""
Create visual comparisons between curated crown coordinates and old census coordinates.

For each selected row in the corrected split shapefiles, this script nearest-joins
the row geometry to the curated crown centroid, then draws:
- a context crop around the current pipeline/curated coordinate,
- a crop at the current pipeline/curated coordinate,
- a crop at the old coordinate stored in the curated attributes.

This is CPU-only and is meant for visual data QA, not model evaluation.
"""

import argparse
import html
import math
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw
from rasterio.warp import transform as rio_transform


def safe_mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_stem(name):
    return os.path.splitext(os.path.basename(str(name).strip()))[0].lower()


def safe_name(value):
    text = str(value).strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)[:80]


def build_tif_index(imagery_root):
    folder_to_paths = {}
    for root, _, files in os.walk(imagery_root):
        tif_files = [f for f in files if f.lower().endswith((".tif", ".tiff"))]
        if not tif_files:
            continue
        rel = os.path.relpath(root, imagery_root)
        parts = rel.split(os.sep)
        folder_key = None
        for part in parts:
            if part.startswith("2023-"):
                folder_key = part
                break
        if folder_key is None:
            folder_key = parts[0] if parts and parts[0] != "." else ""
        folder_to_paths.setdefault(folder_key, [])
        for filename in tif_files:
            folder_to_paths[folder_key].append(os.path.join(root, filename))
    total = sum(len(v) for v in folder_to_paths.values())
    print(f"[INFO] Indexed TIFF folders: {len(folder_to_paths)}")
    print(f"[INFO] Indexed TIFF files  : {total}")
    if total == 0:
        raise RuntimeError(f"No TIFFs found under imagery root: {imagery_root}")
    return folder_to_paths


def resolve_tif_path(folder_to_paths, folder, filename):
    folder = str(folder).strip()
    stem = normalize_stem(filename)
    if folder not in folder_to_paths:
        raise FileNotFoundError(f"Folder key not found in TIFF index: {folder}")

    exact = []
    contains = []
    reverse_contains = []
    for path in folder_to_paths[folder]:
        tif_stem = normalize_stem(path)
        if tif_stem == stem:
            exact.append(path)
        elif stem in tif_stem:
            contains.append(path)
        elif tif_stem in stem:
            reverse_contains.append(path)

    for matches in [exact, contains, reverse_contains]:
        if matches:
            return sorted(matches, key=len)[0]
    raise FileNotFoundError(f"No TIFF match for folder={folder}, file={filename}")


def raster_pixel_size(src):
    x_size = abs(float(src.transform.a)) if src.transform is not None else 1.0
    y_size = abs(float(src.transform.e)) if src.transform is not None else 1.0
    if not np.isfinite(x_size) or x_size <= 0:
        x_size = 1.0
    if not np.isfinite(y_size) or y_size <= 0:
        y_size = x_size
    return x_size, y_size


def world_to_pixel(src, x, y, source_crs):
    x = float(x)
    y = float(y)
    if src.crs is not None and str(src.crs) != str(source_crs):
        xs, ys = rio_transform(source_crs, src.crs, [x], [y])
        x, y = xs[0], ys[0]
    row, col = src.index(x, y)
    return float(col), float(row)


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


def resize_square(img, size):
    return img.resize((size, size), Image.Resampling.BICUBIC)


def draw_cross(draw, x, y, color, radius=12, width=3):
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=width)
    draw.line((x - radius * 1.4, y, x + radius * 1.4, y), fill=color, width=width)
    draw.line((x, y - radius * 1.4, x, y + radius * 1.4), fill=color, width=width)


def draw_text_box(draw, xy, lines, fill=(0, 0, 0, 185), text_fill=(255, 255, 255)):
    x, y = xy
    line_h = 14
    widths = [draw.textlength(str(line)) for line in lines]
    w = int(max(widths) + 14) if widths else 60
    h = int(line_h * len(lines) + 10)
    draw.rounded_rectangle((x, y, x + w, y + h), radius=4, fill=fill)
    for i, line in enumerate(lines):
        draw.text((x + 7, y + 5 + i * line_h), str(line), fill=text_fill)


def label_panel(img, text):
    draw = ImageDraw.Draw(img, "RGBA")
    draw_text_box(draw, (6, 6), [text])


def make_context(src, new_px, new_py, old_px, old_py, old_new_dist_m, args):
    x_size, y_size = raster_pixel_size(src)
    mean_pixel_m = max((x_size + y_size) / 2.0, 1e-6)
    context_m = max(args.context_size_m, min(args.max_context_size_m, old_new_dist_m * 2.4))
    context_px = max(256, int(round(context_m / mean_pixel_m)))

    img = read_patch_from_src(src, new_px, new_py, context_px)
    img = resize_square(img, args.panel_size)
    draw = ImageDraw.Draw(img, "RGBA")
    scale = args.panel_size / context_px

    def to_context(px, py):
        return (
            args.panel_size / 2 + (float(px) - new_px) * scale,
            args.panel_size / 2 + (float(py) - new_py) * scale,
        )

    draw_cross(draw, args.panel_size / 2, args.panel_size / 2, (0, 220, 255, 255))
    ox, oy = to_context(old_px, old_py)
    if 0 <= ox < args.panel_size and 0 <= oy < args.panel_size:
        draw_cross(draw, ox, oy, (255, 45, 85, 255))
        note = "context: cyan=new crown, magenta=old census"
    else:
        note = f"context: cyan=new crown; old census outside ({old_new_dist_m:.1f}m)"
    label_panel(img, note)
    return img


def crop_panel(src, px, py, patch_size_px, panel_size, label, color):
    img = read_patch_from_src(src, px, py, patch_size_px)
    img = resize_square(img, panel_size)
    draw = ImageDraw.Draw(img, "RGBA")
    draw_cross(draw, panel_size / 2, panel_size / 2, color)
    label_panel(img, label)
    return img


def read_splits(split_dir, splits, target_crs):
    frames = []
    for split in splits:
        path = Path(split_dir) / f"valid_points_{split}.shp"
        if not path.exists():
            continue
        gdf = gpd.read_file(path).to_crs(target_crs)
        gdf["split"] = split
        gdf["split_row"] = np.arange(len(gdf))
        frames.append(gdf)
    if not frames:
        raise FileNotFoundError(f"No split shapefiles found in {split_dir}")
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=target_crs)


def join_curated(points, curated_path, args):
    curated = gpd.read_file(curated_path).to_crs(args.target_crs)
    curated = curated.copy()
    curated["curated_species"] = curated[args.curated_species_field].astype(str).str.strip()
    curated["old_east"] = pd.to_numeric(curated[args.old_east_field], errors="coerce")
    curated["old_north"] = pd.to_numeric(curated[args.old_north_field], errors="coerce")
    for optional in ["NOMBRE_CIE", "cod_match", "cod_superv", "PCA"]:
        if optional in curated.columns:
            curated[f"curated_{optional}"] = curated[optional].astype(str)
    keep_cols = [
        "curated_species",
        "old_east",
        "old_north",
        "geometry",
    ] + [c for c in curated.columns if c.startswith("curated_") and c != "curated_species"]
    curated["geometry"] = curated.geometry.centroid
    curated = gpd.GeoDataFrame(curated[keep_cols], geometry="geometry", crs=args.target_crs)

    joined = gpd.sjoin_nearest(
        points,
        curated,
        how="left",
        distance_col="curated_centroid_dist_m",
    )
    joined = (
        joined.sort_values("curated_centroid_dist_m", na_position="last")
        .groupby(level=0, sort=False)
        .first()
        .reindex(points.index)
    )
    return gpd.GeoDataFrame(joined, geometry="geometry", crs=args.target_crs)


def select_rows(df, args):
    out = df.copy()
    if args.only_positive:
        out = out[out[args.binary_field].astype(str).isin(["1", args.target_label])]
    if args.target_label:
        out = out[out[args.tree_field].astype(str).str.strip().str.lower() == args.target_label.strip().lower()]
    out = out[out["old_east"].notna() & out["old_north"].notna()].copy()
    out["old_new_dist_m"] = np.sqrt(
        (out.geometry.x.astype(float) - out["old_east"].astype(float)) ** 2
        + (out.geometry.y.astype(float) - out["old_north"].astype(float)) ** 2
    )
    if args.sort_by == "random":
        out = out.sample(frac=1.0, random_state=args.seed)
    else:
        out = out.sort_values(args.sort_by, ascending=args.ascending)
    return out.head(args.limit)


def relpath(path, start):
    return os.path.relpath(os.path.abspath(path), os.path.abspath(start)).replace(os.sep, "/")


def write_html(rows, output_html):
    cards = []
    for row in rows:
        src = relpath(row["debug_image"], output_html.parent)
        title = html.escape(
            f"{row['split']} row={row['split_row']} species={row['tree']} old_new={row['old_new_dist_m']:.1f}m"
        )
        caption = html.escape(
            f"{row['split']} row {row['split_row']} | {row['tree']} | old-new {row['old_new_dist_m']:.1f}m"
        )
        cards.append(
            f'<figure title="{title}"><img src="{html.escape(src)}" loading="lazy" />'
            f"<figcaption>{caption}</figcaption></figure>"
        )

    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Coordinate Comparison Patches</title>
  <style>
    body {{ margin: 24px; font-family: Arial, sans-serif; background: #f7f8fa; color: #202124; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .meta {{ color: #667085; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(520px, 1fr)); gap: 14px; }}
    figure {{ margin: 0; padding: 8px; background: white; border: 1px solid #d7dbe3; border-radius: 6px; }}
    img {{ width: 100%; display: block; }}
    figcaption {{ margin-top: 6px; font-size: 12px; color: #475467; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  </style>
</head>
<body>
  <h1>Coordinate Comparison Patches</h1>
  <div class="meta">cyan=new curated/pipeline coordinate; magenta=old census coordinate</div>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    output_html.write_text(page, encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Make visual patches comparing curated and old census coordinates.")
    p.add_argument("--split_dir", default="./outputs/splits_binary_curated")
    p.add_argument("--curated", required=True)
    p.add_argument("--imagery_root", default="/mnt/parscratch/users/aca21jo/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023")
    p.add_argument("--output_dir", default="./outputs/coordinate_comparison_patches")
    p.add_argument("--splits", nargs="+", default=["val"])
    p.add_argument("--target_crs", default="EPSG:32718")
    p.add_argument("--tree_field", default="Tree")
    p.add_argument("--binary_field", default="BinaryTree")
    p.add_argument("--target_label", default="Shihuahuaco")
    p.add_argument("--curated_species_field", default="NOMBRE_COM")
    p.add_argument("--old_east_field", default="COORDENADA")
    p.add_argument("--old_north_field", default="COORDENA_1")
    p.add_argument("--folder_field", default="Folder")
    p.add_argument("--file_field", default="File")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sort_by", default="old_new_dist_m", choices=["old_new_dist_m", "random", "split_row"])
    p.add_argument("--ascending", action="store_true")
    p.add_argument("--only_positive", action="store_true")
    p.add_argument("--patch_size_px", type=int, default=160)
    p.add_argument("--context_size_m", type=float, default=80.0)
    p.add_argument("--max_context_size_m", type=float, default=250.0)
    p.add_argument("--panel_size", type=int, default=224)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    safe_mkdir(image_dir)

    points = read_splits(args.split_dir, args.splits, args.target_crs)
    joined = join_curated(points, args.curated, args)
    selected = select_rows(joined, args)

    folder_to_paths = build_tif_index(args.imagery_root)
    rows = []
    for n, (_, row) in enumerate(selected.iterrows(), start=1):
        try:
            image_path = resolve_tif_path(folder_to_paths, row[args.folder_field], row[args.file_field])
            with rasterio.open(image_path) as src:
                new_px, new_py = world_to_pixel(src, row.geometry.x, row.geometry.y, args.target_crs)
                old_px, old_py = world_to_pixel(src, row["old_east"], row["old_north"], args.target_crs)
                context = make_context(src, new_px, new_py, old_px, old_py, row["old_new_dist_m"], args)
                new_crop = crop_panel(
                    src, new_px, new_py, args.patch_size_px, args.panel_size,
                    "new curated/pipeline", (0, 220, 255, 255)
                )
                old_crop = crop_panel(
                    src, old_px, old_py, args.patch_size_px, args.panel_size,
                    "old census coordinate", (255, 45, 85, 255)
                )

            gap = 10
            header_h = 78
            width = args.panel_size * 3 + gap * 2
            height = args.panel_size + header_h
            canvas = Image.new("RGB", (width, height), (245, 247, 250))
            draw = ImageDraw.Draw(canvas, "RGBA")
            header = [
                f"{row['split']} row={int(row['split_row'])} tree={row[args.tree_field]} curated={row['curated_species']}",
                f"old-new distance={float(row['old_new_dist_m']):.1f}m; curated centroid match={float(row['curated_centroid_dist_m']):.3f}m",
                f"{row[args.folder_field]} | {row[args.file_field]}",
            ]
            draw_text_box(draw, (8, 8), header, fill=(16, 24, 40, 225))
            y0 = header_h
            canvas.paste(context, (0, y0))
            canvas.paste(new_crop, (args.panel_size + gap, y0))
            canvas.paste(old_crop, (args.panel_size * 2 + gap * 2, y0))
            out_img = image_dir / f"{n:04d}_{row['split']}_{int(row['split_row'])}_{safe_name(row[args.tree_field])}.png"
            canvas.save(out_img)

            out_row = {
                "split": row["split"],
                "split_row": int(row["split_row"]),
                "tree": str(row[args.tree_field]),
                "curated_species": str(row["curated_species"]),
                "old_new_dist_m": float(row["old_new_dist_m"]),
                "curated_centroid_dist_m": float(row["curated_centroid_dist_m"]),
                "folder": str(row[args.folder_field]),
                "file": str(row[args.file_field]),
                "image_path": image_path,
                "debug_image": str(out_img),
                "new_x": float(row.geometry.x),
                "new_y": float(row.geometry.y),
                "old_east": float(row["old_east"]),
                "old_north": float(row["old_north"]),
            }
            rows.append(out_row)
        except Exception as exc:
            print(f"[WARN] Failed row split={row.get('split')} split_row={row.get('split_row')}: {exc}")

    csv_path = output_dir / "coordinate_comparison_patches.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    html_path = output_dir / "contact_sheet.html"
    write_html(rows, html_path)
    print(html_path)
    print(csv_path)


if __name__ == "__main__":
    main()
