"""
Create visual debug sheets from MIL realignment CSV outputs.

This reads the CSVs produced by apply_mil_realign.py, such as
realigned_points.csv or realigned_points_all_models.csv. Each card shows:
- a context crop around the original point,
- the original point crop,
- the raw-instance correction crop,
- the context-head correction crop.

The script is intended for quick visual QA of weak-census/generalisation runs.
"""

import argparse
import html
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw


def safe_mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_name(value):
    text = str(value).strip()
    out = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return out[:120] or "model"


def raster_pixel_size(src):
    x_size = abs(float(src.transform.a)) if src.transform is not None else 1.0
    y_size = abs(float(src.transform.e)) if src.transform is not None else 1.0
    if not np.isfinite(x_size) or x_size <= 0:
        x_size = 1.0
    if not np.isfinite(y_size) or y_size <= 0:
        y_size = x_size
    return x_size, y_size


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
    draw.line((x - radius * 1.45, y, x + radius * 1.45, y), fill=color, width=width)
    draw.line((x, y - radius * 1.45, x, y + radius * 1.45), fill=color, width=width)


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


def row_float(row, key, default=0.0):
    value = row.get(key, default)
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def correction_points(row):
    center_px = row_float(row, "center_px")
    center_py = row_float(row, "center_py")
    raw_px = center_px + row_float(row, "raw_dx_px")
    raw_py = center_py + row_float(row, "raw_dy_px")
    context_px = center_px + row_float(row, "context_dx_px")
    context_py = center_py + row_float(row, "context_dy_px")
    selected_px = center_px + row_float(row, "selected_dx_px")
    selected_py = center_py + row_float(row, "selected_dy_px")
    return {
        "original": (center_px, center_py),
        "raw": (raw_px, raw_py),
        "context": (context_px, context_py),
        "selected": (selected_px, selected_py),
    }


def context_panel(src, row, args):
    points = correction_points(row)
    center_px, center_py = points["original"]
    max_offset_m = max(
        row_float(row, "raw_offset_m"),
        row_float(row, "context_offset_m"),
        row_float(row, "selected_offset_m"),
    )
    x_size, y_size = raster_pixel_size(src)
    mean_pixel_m = max((x_size + y_size) / 2.0, 1e-6)
    context_m = max(args.context_size_m, min(args.max_context_size_m, max_offset_m * 2.6))
    context_px = max(args.panel_size, int(round(context_m / mean_pixel_m)))

    img = read_patch_from_src(src, center_px, center_py, context_px)
    img = resize_square(img, args.panel_size)
    draw = ImageDraw.Draw(img, "RGBA")
    scale = args.panel_size / context_px

    def to_context(px, py):
        return (
            args.panel_size / 2 + (float(px) - center_px) * scale,
            args.panel_size / 2 + (float(py) - center_py) * scale,
        )

    colors = {
        "original": (255, 45, 85, 255),
        "raw": (0, 220, 255, 255),
        "context": (255, 132, 0, 255),
        "selected": (255, 230, 0, 255),
    }
    for name in ["raw", "context", "selected", "original"]:
        x, y = to_context(*points[name])
        if 0 <= x < args.panel_size and 0 <= y < args.panel_size:
            draw_cross(draw, x, y, colors[name], radius=10 if name != "original" else 13, width=3)

    label_panel(img, "context: magenta=original cyan=raw orange=context yellow=selected")
    return img


def crop_panel(src, px, py, patch_size_px, panel_size, label, color):
    img = read_patch_from_src(src, px, py, patch_size_px)
    img = resize_square(img, panel_size)
    draw = ImageDraw.Draw(img, "RGBA")
    draw_cross(draw, panel_size / 2, panel_size / 2, color, radius=13, width=3)
    label_panel(img, label)
    return img


def compact_file(row):
    folder = str(row.get("Folder", row.get("folder", "")))
    filename = str(row.get("File", row.get("file", "")))
    if len(filename) > 44:
        filename = filename[:41] + "..."
    return f"{folder} | {filename}".strip(" |")


def compose_debug_image(row, args, out_path):
    image_path = str(row["image_path"])
    points = correction_points(row)
    patch_size_px = int(args.patch_size_px)
    if patch_size_px <= 0:
        patch_size_px = int(row_float(row, "patch_size_px", 160))
    with rasterio.open(image_path) as src:
        context = context_panel(src, row, args)
        original = crop_panel(
            src,
            *points["original"],
            patch_size_px,
            args.panel_size,
            "original point",
            (255, 45, 85, 255),
        )
        raw = crop_panel(
            src,
            *points["raw"],
            patch_size_px,
            args.panel_size,
            "raw correction",
            (0, 220, 255, 255),
        )
        context_crop = crop_panel(
            src,
            *points["context"],
            patch_size_px,
            args.panel_size,
            "context correction",
            (255, 132, 0, 255),
        )

    gap = 10
    header_h = 86
    w = args.panel_size * 4 + gap * 3
    h = args.panel_size + header_h
    canvas = Image.new("RGB", (w, h), (245, 247, 250))
    draw = ImageDraw.Draw(canvas, "RGBA")

    model = str(row.get("model_run", "model"))
    lines = [
        f"{model}",
        (
            f"bag={row_float(row, 'bag_prob_1'):.3f} pred={int(row_float(row, 'is_positive_at_threshold'))} "
            f"raw={row_float(row, 'raw_selected_prob_1'):.3f}/{row_float(row, 'raw_offset_m'):.1f}m "
            f"ctx={row_float(row, 'context_selected_prob_1'):.3f}/{row_float(row, 'context_offset_m'):.1f}m "
            f"sel={row_float(row, 'selected_offset_m'):.1f}m"
        ),
        compact_file(row),
    ]
    draw_text_box(draw, (8, 8), lines, fill=(16, 24, 40, 225))

    y0 = header_h
    x = 0
    for img in [context, original, raw, context_crop]:
        canvas.paste(img, (x, y0))
        x += args.panel_size + gap
    canvas.save(out_path)


def relpath(path, start):
    return os.path.relpath(os.path.abspath(path), os.path.abspath(start)).replace(os.sep, "/")


def write_html(rows, output_html, title):
    cards = []
    for row in rows:
        src = relpath(row["debug_image"], output_html.parent)
        title_attr = html.escape(
            f"{row.get('model_run', '')} bag={row_float(row, 'bag_prob_1'):.3f} "
            f"raw={row_float(row, 'raw_selected_prob_1'):.3f} "
            f"ctx={row_float(row, 'context_selected_prob_1'):.3f} "
            f"selected_offset={row_float(row, 'selected_offset_m'):.1f}m"
        )
        caption = html.escape(
            f"{int(row.get('debug_rank', 0)):03d} | bag {row_float(row, 'bag_prob_1'):.3f} | "
            f"raw {row_float(row, 'raw_selected_prob_1'):.3f}/{row_float(row, 'raw_offset_m'):.1f}m | "
            f"ctx {row_float(row, 'context_selected_prob_1'):.3f}/{row_float(row, 'context_offset_m'):.1f}m | "
            f"sel {row_float(row, 'selected_offset_m'):.1f}m"
        )
        cards.append(
            f'<figure title="{title_attr}"><img src="{html.escape(src)}" loading="lazy" />'
            f"<figcaption>{caption}</figcaption></figure>"
        )

    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 24px;
      font-family: Arial, sans-serif;
      color: #202124;
      background: #f7f8fa;
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .meta {{ color: #667085; margin-bottom: 18px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(720px, 1fr));
      gap: 14px;
    }}
    figure {{
      margin: 0;
      padding: 8px;
      background: white;
      border: 1px solid #d7dbe3;
      border-radius: 6px;
    }}
    img {{ width: 100%; display: block; }}
    figcaption {{
      margin-top: 6px;
      font-size: 12px;
      color: #475467;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="meta">{len(rows)} shown</div>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    output_html.write_text(page, encoding="utf-8")


def select_rows(df, args):
    out = df.copy()
    if args.only_positive_threshold:
        out = out[pd.to_numeric(out["is_positive_at_threshold"], errors="coerce").fillna(0) == 1]
    if args.min_bag_prob is not None:
        out = out[pd.to_numeric(out["bag_prob_1"], errors="coerce").fillna(-1) >= args.min_bag_prob]
    if args.min_offset_m is not None:
        out = out[pd.to_numeric(out["selected_offset_m"], errors="coerce").fillna(0) >= args.min_offset_m]
    if args.require_raw_context_disagree:
        out = out[pd.to_numeric(out["selection_disagrees_with_context"], errors="coerce").fillna(0) == 1]

    if args.sort_by == "random":
        out = out.sample(frac=1.0, random_state=args.seed)
    else:
        if args.sort_by not in out.columns:
            raise ValueError(f"sort column not found: {args.sort_by}")
        sort_values = pd.to_numeric(out[args.sort_by], errors="coerce")
        out = out.assign(_sort_values=sort_values).sort_values("_sort_values", ascending=args.ascending)
        out = out.drop(columns=["_sort_values"])
    if args.limit_per_model <= 0:
        return out
    return out.head(args.limit_per_model)


def write_index(model_pages, output_dir):
    rows = []
    for model, page_path, count in model_pages:
        href = relpath(page_path, output_dir)
        rows.append(
            f'<li><a href="{html.escape(href)}">{html.escape(model)}</a> '
            f'<span>{count} shown</span></li>'
        )
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Realignment Debug Index</title>
  <style>
    body {{ margin: 24px; font-family: Arial, sans-serif; background: #f7f8fa; color: #202124; }}
    li {{ margin: 8px 0; }}
    span {{ color: #667085; }}
  </style>
</head>
<body>
  <h1>Realignment Debug Index</h1>
  <ul>{''.join(rows)}</ul>
</body>
</html>
"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Create debug contact sheets from MIL realignment CSVs.")
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--models", nargs="*", default=None, help="Optional model_run names to include.")
    p.add_argument("--limit_per_model", type=int, default=40, help="Rows per model. Use 0 or negative for all rows.")
    p.add_argument("--sort_by", default="bag_prob_1")
    p.add_argument("--ascending", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only_positive_threshold", action="store_true")
    p.add_argument("--min_bag_prob", type=float, default=None)
    p.add_argument("--min_offset_m", type=float, default=None)
    p.add_argument("--require_raw_context_disagree", action="store_true")
    p.add_argument("--patch_size_px", type=int, default=160)
    p.add_argument("--context_size_m", type=float, default=55.0)
    p.add_argument("--max_context_size_m", type=float, default=90.0)
    p.add_argument("--panel_size", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    safe_mkdir(output_dir)

    df = pd.read_csv(input_csv)
    required = ["image_path", "center_px", "center_py", "bag_prob_1"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {missing}")
    if "model_run" not in df.columns:
        df["model_run"] = input_csv.parent.name

    if args.models:
        df = df[df["model_run"].astype(str).isin(args.models)].copy()
    if df.empty:
        raise ValueError("No rows left after model filtering.")

    model_pages = []
    for model, group in df.groupby("model_run", sort=True):
        selected = select_rows(group, args).copy()
        model_dir = output_dir / safe_name(model)
        image_dir = model_dir / "images"
        safe_mkdir(image_dir)

        rows = []
        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            row = row.to_dict()
            row["debug_rank"] = rank
            out_path = image_dir / f"{rank:04d}_{safe_name(model)}.png"
            try:
                compose_debug_image(row, args, out_path)
            except Exception as exc:
                print(f"[WARN] failed row rank={rank} model={model}: {exc}")
                continue
            row["debug_image"] = str(out_path)
            rows.append(row)

        page_path = model_dir / "contact_sheet.html"
        write_html(rows, page_path, title=f"Realignment Debug - {model}")
        model_pages.append((str(model), page_path, len(rows)))
        print(page_path)

    write_index(model_pages, output_dir)
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
