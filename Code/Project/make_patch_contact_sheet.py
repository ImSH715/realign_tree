import argparse
import html
import os
from pathlib import Path

import pandas as pd
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Build an HTML contact sheet for debug patch QA.")
    p.add_argument("--debug_dir", required=True, help="Directory containing debug_patches.csv and PNGs.")
    p.add_argument("--output_html", default=None)
    p.add_argument("--thumb_size", type=int, default=180)
    p.add_argument("--max_per_class", type=int, default=80)
    p.add_argument("--no_crosshair", action="store_true")
    return p.parse_args()


def relpath(path, start):
    return os.path.relpath(os.path.abspath(path), os.path.abspath(start)).replace(os.sep, "/")


def resolve_patch_path(raw_path, debug_dir):
    patch_path = Path(str(raw_path))
    if patch_path.is_absolute():
        return patch_path

    cwd_path = Path.cwd() / patch_path
    if cwd_path.exists():
        return cwd_path

    debug_path = debug_dir / patch_path
    if debug_path.exists():
        return debug_path

    name_path = debug_dir / patch_path.name
    if name_path.exists():
        return name_path

    return cwd_path


def patch_quality(path):
    metrics = {
        "black_fraction": None,
        "bright_fraction": None,
        "flags": [],
    }
    try:
        img = Image.open(path).convert("RGB")
        pixels = list(img.getdata())
    except Exception as e:
        metrics["flags"].append(f"unreadable: {e}")
        return metrics

    if not pixels:
        metrics["flags"].append("empty")
        return metrics

    n = len(pixels)
    black = sum(1 for r, g, b in pixels if max(r, g, b) <= 5)
    bright = sum(1 for r, g, b in pixels if min(r, g, b) >= 245)
    metrics["black_fraction"] = black / n
    metrics["bright_fraction"] = bright / n

    if metrics["black_fraction"] >= 0.25:
        metrics["flags"].append("high black/no-data")
    if metrics["bright_fraction"] >= 0.25:
        metrics["flags"].append("high bright/washed")

    return metrics


def main():
    args = parse_args()
    debug_dir = Path(args.debug_dir)
    csv_path = debug_dir / "debug_patches.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing debug patch CSV: {csv_path}")

    output_html = Path(args.output_html) if args.output_html else debug_dir / "contact_sheet.html"
    output_html.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "label" not in df.columns or "debug_patch" not in df.columns:
        raise ValueError(f"Expected label/debug_patch columns. Available: {df.columns.tolist()}")

    rows = []
    for label, group in df.groupby("label", sort=True):
        clean = group[group["debug_patch"].notna()].head(args.max_per_class)
        rows.append(f"<h2>Class {html.escape(str(label))} <span>{len(clean)} shown / {len(group)} listed</span></h2>")
        rows.append('<div class="grid">')
        for _, row in clean.iterrows():
            patch_path = resolve_patch_path(row["debug_patch"], debug_dir)
            src = relpath(patch_path, output_html.parent)
            title_bits = [
                f"idx={row.get('idx', '')}",
                f"label={row.get('label', '')}",
                f"folder={row.get('folder', '')}",
                f"file={row.get('file', '')}",
                f"mode={row.get('coord_mode_used', '')}",
                f"px={row.get('pixel_x', '')}",
                f"py={row.get('pixel_y', '')}",
            ]
            quality = patch_quality(patch_path)
            if quality["black_fraction"] is not None:
                title_bits.append(f"black={quality['black_fraction']:.3f}")
                title_bits.append(f"bright={quality['bright_fraction']:.3f}")
            if quality["flags"]:
                title_bits.append("flags=" + "; ".join(quality["flags"]))
            title = html.escape(" | ".join(str(x) for x in title_bits))
            caption = html.escape(f"{row.get('idx', '')} - {row.get('folder', '')}")
            flag_html = ""
            if quality["flags"]:
                flag_html = '<div class="flag">' + html.escape(", ".join(quality["flags"])) + "</div>"
            crosshair_class = "" if args.no_crosshair else " with-crosshair"
            rows.append(
                f'<figure title="{title}">'
                f'<div class="thumb{crosshair_class}"><img src="{html.escape(src)}" loading="lazy" /></div>'
                f"<figcaption>{caption}</figcaption>"
                f"{flag_html}"
                "</figure>"
            )
        rows.append("</div>")

    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Debug Patch Contact Sheet</title>
  <style>
    body {{
      margin: 24px;
      font-family: Arial, sans-serif;
      color: #202124;
      background: #f7f8fa;
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; }}
    h2 span {{ color: #667085; font-weight: normal; font-size: 14px; }}
    .meta {{ color: #667085; margin-bottom: 18px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax({args.thumb_size}px, 1fr));
      gap: 12px;
    }}
    figure {{
      margin: 0;
      padding: 8px;
      background: white;
      border: 1px solid #d7dbe3;
      border-radius: 6px;
    }}
    .thumb {{
      position: relative;
    }}
    .thumb.with-crosshair::before,
    .thumb.with-crosshair::after {{
      content: "";
      position: absolute;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
      pointer-events: none;
      z-index: 2;
    }}
    .thumb.with-crosshair::before {{
      width: 22px;
      height: 22px;
      border: 2px solid #ff2d55;
      border-radius: 50%;
      box-shadow: 0 0 0 1px white;
    }}
    .thumb.with-crosshair::after {{
      width: 34px;
      height: 34px;
      background:
        linear-gradient(#ff2d55, #ff2d55) center / 34px 2px no-repeat,
        linear-gradient(#ff2d55, #ff2d55) center / 2px 34px no-repeat;
      filter: drop-shadow(0 0 1px white);
      opacity: 0.9;
    }}
    img {{
      width: 100%;
      aspect-ratio: 1;
      object-fit: cover;
      display: block;
      image-rendering: auto;
      border: 1px solid #eef0f4;
    }}
    figcaption {{
      margin-top: 6px;
      font-size: 12px;
      color: #475467;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .flag {{
      margin-top: 4px;
      color: #b42318;
      font-size: 12px;
      font-weight: bold;
    }}
  </style>
</head>
<body>
  <h1>Debug Patch Contact Sheet</h1>
  <div class="meta">{html.escape(str(csv_path))}</div>
  {''.join(rows)}
</body>
</html>
"""

    output_html.write_text(page, encoding="utf-8")
    print(output_html)


if __name__ == "__main__":
    main()
