"""
Create visual debug sheets for MIL corrections.

Input is the mil_instance_pca.csv produced by analyze_mil_feature_space.py.
Each debug image shows:
- a larger context crop with the original point, all candidate positions, and
  the selected raw-instance correction,
- the original centered patch,
- the selected corrected patch.
"""

import argparse
import html
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont


def safe_mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


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


def draw_cross(draw, x, y, color, radius=11, width=3):
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=width)
    draw.line((x - radius * 1.45, y, x + radius * 1.45, y), fill=color, width=width)
    draw.line((x, y - radius * 1.45, x, y + radius * 1.45), fill=color, width=width)


def draw_text_box(draw, xy, lines, fill=(0, 0, 0, 185), text_fill=(255, 255, 255), font=None):
    x, y = xy
    line_h = 14
    widths = [draw.textlength(str(line), font=font) for line in lines]
    w = int(max(widths) + 14) if widths else 60
    h = int(line_h * len(lines) + 10)
    draw.rounded_rectangle((x, y, x + w, y + h), radius=4, fill=fill)
    for i, line in enumerate(lines):
        draw.text((x + 7, y + 5 + i * line_h), str(line), fill=text_fill, font=font)


def panel_label(img, text):
    draw = ImageDraw.Draw(img, "RGBA")
    draw_text_box(draw, (6, 6), [text])


def resize_square(img, size):
    return img.resize((size, size), Image.Resampling.BICUBIC)


def context_panel(src, group, selected, context_size_m, output_size):
    center_px = float(selected["center_px"])
    center_py = float(selected["center_py"])
    best_px = float(selected["px"])
    best_py = float(selected["py"])
    x_size, y_size = raster_pixel_size(src)
    patch_size_px = int(round(context_size_m / max((x_size + y_size) / 2.0, 1e-6)))
    patch_size_px = max(256, patch_size_px)
    img = read_patch_from_src(src, center_px, center_py, patch_size_px)
    img = resize_square(img, output_size)

    scale_x = output_size / patch_size_px
    scale_y = output_size / patch_size_px
    draw = ImageDraw.Draw(img, "RGBA")

    def to_context(px, py):
        return (
            output_size / 2 + (float(px) - center_px) * scale_x,
            output_size / 2 + (float(py) - center_py) * scale_y,
        )

    for _, row in group.iterrows():
        x, y = to_context(row["px"], row["py"])
        prob = float(row.get("instance_prob_1", 0.0))
        alpha = int(75 + 180 * max(0.0, min(1.0, prob)))
        r = 3 if int(row.get("is_selected", 0)) == 0 else 5
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 210, 0, alpha), outline=(0, 0, 0, 160))

    if "is_context_selected" in group.columns:
        context_selected = group[group["is_context_selected"] == 1]
        for _, row in context_selected.iterrows():
            x, y = to_context(row["px"], row["py"])
            draw_cross(draw, x, y, (255, 132, 0, 255), radius=10, width=2)

    draw_cross(draw, output_size / 2, output_size / 2, (255, 45, 85, 255), radius=13, width=3)
    bx, by = to_context(best_px, best_py)
    draw_cross(draw, bx, by, (0, 220, 255, 255), radius=13, width=3)
    panel_label(img, "context: magenta=original cyan=raw selected orange=context selected")
    return img


def crop_panel(src, px, py, patch_size_px, output_size, label, color):
    img = read_patch_from_src(src, px, py, patch_size_px)
    img = resize_square(img, output_size)
    draw = ImageDraw.Draw(img, "RGBA")
    draw_cross(draw, output_size / 2, output_size / 2, color, radius=13, width=3)
    panel_label(img, label)
    return img


def compose_debug_image(group, selected, args, out_path):
    image_path = str(selected["image_path"])
    patch_size_px = int(args.patch_size_px) if args.patch_size_px > 0 else int(selected.get("patch_size_px", 224))
    with rasterio.open(image_path) as src:
        context = context_panel(src, group, selected, args.context_size_m, args.panel_size)
        original = crop_panel(
            src,
            float(selected["center_px"]),
            float(selected["center_py"]),
            patch_size_px,
            args.panel_size,
            "original point",
            (255, 45, 85, 255),
        )
        corrected = crop_panel(
            src,
            float(selected["px"]),
            float(selected["py"]),
            patch_size_px,
            args.panel_size,
            "selected correction",
            (0, 220, 255, 255),
        )

    gap = 10
    header_h = 76
    w = args.panel_size * 3 + gap * 2
    h = args.panel_size + header_h
    canvas = Image.new("RGB", (w, h), (245, 247, 250))
    draw = ImageDraw.Draw(canvas, "RGBA")

    offset_m = float(selected.get("offset_m", 0.0))
    lines = [
        f"bag={int(selected['bag_index'])} true={int(selected['y_true'])} pred={int(selected['y_pred'])} status={selected.get('status', '')}",
        (
            f"bag_prob={float(selected['bag_prob_1']):.3f} "
            f"raw_sel={float(selected.get('selected_raw_instance_prob_1', selected.get('selected_instance_prob_1', 0.0))):.3f} "
            f"ctx_at_sel={float(selected.get('selected_instance_prob_1', 0.0)):.3f} "
            f"ctx_best={float(selected.get('selected_context_instance_prob_1', 0.0)):.3f} "
            f"offset={offset_m:.1f}m"
        ),
        f"{selected.get('folder', '')} | {selected.get('file', '')}",
    ]
    draw_text_box(draw, (8, 8), lines, fill=(16, 24, 40, 225))

    y0 = header_h
    canvas.paste(context, (0, y0))
    canvas.paste(original, (args.panel_size + gap, y0))
    canvas.paste(corrected, (args.panel_size * 2 + gap * 2, y0))
    canvas.save(out_path)


def choose_bags(selected_df, args):
    df = selected_df.copy()
    if args.status:
        df = df[df["status"].astype(str) == args.status]
    if args.true_label != "":
        df = df[df["y_true"].astype(str) == str(args.true_label)]
    if args.pred_label != "":
        df = df[df["y_pred"].astype(str) == str(args.pred_label)]

    if args.sort_by == "random":
        df = df.sample(frac=1.0, random_state=args.seed)
    else:
        ascending = args.ascending
        df = df.sort_values(args.sort_by, ascending=ascending)
    return df.head(args.limit)


def relpath(path, start):
    return os.path.relpath(os.path.abspath(path), os.path.abspath(start)).replace(os.sep, "/")


def write_html(rows, output_html, image_dir, args):
    cards = []
    for row in rows:
        src = relpath(row["debug_image"], output_html.parent)
        title = html.escape(
            f"bag={row['bag_index']} true={row['y_true']} pred={row['y_pred']} "
            f"bag_prob={row['bag_prob_1']:.3f} "
            f"raw_sel={row.get('selected_raw_instance_prob_1', row['selected_instance_prob_1']):.3f} "
            f"ctx_at_sel={row['selected_instance_prob_1']:.3f} "
            f"offset={row['offset_m']:.1f}m"
        )
        caption = html.escape(
            f"bag {int(row['bag_index'])} | {row['status']} | bag {row['bag_prob_1']:.3f} | "
            f"raw {row.get('selected_raw_instance_prob_1', row['selected_instance_prob_1']):.3f} | "
            f"ctx {row['selected_instance_prob_1']:.3f} | {row['offset_m']:.1f}m"
        )
        cards.append(
            f'<figure title="{title}"><img src="{html.escape(src)}" loading="lazy" />'
            f"<figcaption>{caption}</figcaption></figure>"
        )

    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>MIL Debug Patches</title>
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
      grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
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
  <h1>MIL Debug Patches</h1>
  <div class="meta">{html.escape(str(image_dir))}</div>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    output_html.write_text(page, encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Create visual debug sheets for MIL corrections.")
    p.add_argument("--instance_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--limit", type=int, default=80)
    p.add_argument("--sort_by", default="bag_prob_1")
    p.add_argument("--ascending", action="store_true")
    p.add_argument("--status", default="", choices=["", "TP", "TN", "FP", "FN"])
    p.add_argument("--true_label", default="")
    p.add_argument("--pred_label", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patch_size_px", type=int, default=0, help="Use 0 to read patch_size_px from the instance CSV.")
    p.add_argument("--context_size_m", type=float, default=50.0)
    p.add_argument("--panel_size", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    safe_mkdir(image_dir)

    df = pd.read_csv(args.instance_csv)
    required = {"bag_index", "is_selected", "image_path", "center_px", "center_py", "px", "py"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in instance CSV: {sorted(missing)}")

    selected = df[df["is_selected"] == 1].copy()
    if "offset_m" not in selected.columns:
        selected["offset_m"] = np.sqrt(selected["dx_m"] ** 2 + selected["dy_m"] ** 2)
    chosen = choose_bags(selected, args)

    rows = []
    for _, sel in chosen.iterrows():
        bag_id = int(sel["bag_index"])
        group = df[df["bag_index"] == bag_id].copy()
        out_path = image_dir / f"bag_{bag_id:04d}_{sel.get('status', '')}.png"
        try:
            compose_debug_image(group, sel, args, out_path)
            rec = sel.to_dict()
            rec["debug_image"] = str(out_path)
            rows.append(rec)
        except Exception as e:
            print(f"[WARN] Failed bag {bag_id}: {e}")

    debug_csv = output_dir / "mil_debug_patches.csv"
    pd.DataFrame(rows).to_csv(debug_csv, index=False)
    output_html = output_dir / "contact_sheet.html"
    write_html(rows, output_html, image_dir, args)
    print(output_html)
    print(debug_csv)


if __name__ == "__main__":
    main()
