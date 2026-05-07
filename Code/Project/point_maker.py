import os
import argparse
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from PIL import Image
import matplotlib.pyplot as plt


def normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    if data.ndim != 3:
        raise ValueError(f"Expected 3D array (bands, h, w), got {data.shape}")

    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    elif data.shape[0] >= 3:
        data = data[:3]
    else:
        raise ValueError(f"Unsupported band count: {data.shape[0]}")

    rgb = np.transpose(data, (1, 2, 0)).astype(np.float32)
    lo = np.nanpercentile(rgb, 1)
    hi = np.nanpercentile(rgb, 99)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    rgb = (rgb * 255).astype(np.uint8)
    return rgb


def read_patch(src, col_center: float, row_center: float, patch_size: int) -> np.ndarray:
    half = patch_size // 2
    left = int(round(col_center)) - half
    top = int(round(row_center)) - half
    window = Window(left, top, patch_size, patch_size)
    data = src.read(window=window, boundless=True, fill_value=0)
    return normalize_to_uint8(data)


def read_overview(src, cols, rows, margin_px: int):
    col_min = int(min(cols) - margin_px)
    col_max = int(max(cols) + margin_px)
    row_min = int(min(rows) - margin_px)
    row_max = int(max(rows) + margin_px)

    width = col_max - col_min
    height = row_max - row_min

    window = Window(col_min, row_min, width, height)
    data = src.read(window=window, boundless=True, fill_value=0)
    rgb = normalize_to_uint8(data)
    return rgb, col_min, row_min


def safe_name(x: str) -> str:
    s = str(x)
    for ch in ["/", "\\", " ", ":", ";", ",", "(", ")", "[", "]", "{", "}", "'","\""]:
        s = s.replace(ch, "_")
    return s


def save_one_point(row, output_dir: str, patch_size: int, overview_margin_px: int):
    point_id = str(row["point_id"])
    image_path = row["image_path"] if "image_path" in row.index else row["matched_tif"]

    orig_e = float(row["original_east"])
    orig_n = float(row["original_north"])
    ref_e = float(row["refined_east"])
    ref_n = float(row["refined_north"])

    has_gt = ("gt_east" in row.index) and ("gt_north" in row.index)
    if has_gt:
        gt_e = float(row["gt_east"])
        gt_n = float(row["gt_north"])

    with rasterio.open(image_path) as src:
        orig_row, orig_col = src.index(orig_e, orig_n)
        ref_row, ref_col = src.index(ref_e, ref_n)

        if has_gt:
            gt_row, gt_col = src.index(gt_e, gt_n)

        original_patch = read_patch(src, orig_col, orig_row, patch_size)
        refined_patch = read_patch(src, ref_col, ref_row, patch_size)

        cols = [orig_col, ref_col]
        rows = [orig_row, ref_row]
        if has_gt:
            cols.append(gt_col)
            rows.append(gt_row)

        overview_rgb, col_min, row_min = read_overview(src, cols, rows, overview_margin_px)

        orig_x_local = orig_col - col_min
        orig_y_local = orig_row - row_min
        ref_x_local = ref_col - col_min
        ref_y_local = ref_row - row_min

        if has_gt:
            gt_x_local = gt_col - col_min
            gt_y_local = gt_row - row_min

    base = safe_name(point_id)

    orig_patch_path = os.path.join(output_dir, f"{base}_original_patch.png")
    ref_patch_path = os.path.join(output_dir, f"{base}_refined_patch.png")
    overview_path = os.path.join(output_dir, f"{base}_overview.png")
    combined_path = os.path.join(output_dir, f"{base}_combined.png")

    Image.fromarray(original_patch).save(orig_patch_path)
    Image.fromarray(refined_patch).save(ref_patch_path)

    plt.figure(figsize=(8, 8))
    plt.imshow(overview_rgb)
    plt.scatter(orig_x_local, orig_y_local, s=120, marker="o", label="original")
    plt.scatter(ref_x_local, ref_y_local, s=120, marker="x", label="refined")
    if has_gt:
        plt.scatter(gt_x_local, gt_y_local, s=120, marker="^", label="gt")
    plt.legend()
    plt.title(f"Overview | point_id={point_id}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(overview_path, dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(original_patch)
    axes[0].scatter(patch_size // 2, patch_size // 2, s=80, marker="o")
    axes[0].set_title("Original point patch")
    axes[0].axis("off")

    axes[1].imshow(refined_patch)
    axes[1].scatter(patch_size // 2, patch_size // 2, s=80, marker="x")
    axes[1].set_title("Refined point patch")
    axes[1].axis("off")

    axes[2].imshow(overview_rgb)
    axes[2].scatter(orig_x_local, orig_y_local, s=120, marker="o", label="original")
    axes[2].scatter(ref_x_local, ref_y_local, s=120, marker="x", label="refined")
    if has_gt:
        axes[2].scatter(gt_x_local, gt_y_local, s=120, marker="^", label="gt")
    axes[2].legend()
    axes[2].set_title("Overview")
    axes[2].axis("off")

    fig.suptitle(f"point_id={point_id}", fontsize=14)
    plt.tight_layout()
    plt.savefig(combined_path, dpi=200)
    plt.close()

    return {
        "point_id": point_id,
        "original_patch": orig_patch_path,
        "refined_patch": ref_patch_path,
        "overview": overview_path,
        "combined": combined_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--patch_size_px", type=int, default=224)
    parser.add_argument("--overview_margin_px", type=int, default=350)
    parser.add_argument("--max_points", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv).copy()
    df["point_id"] = df["point_id"].astype(str)

    if args.max_points > 0:
        df = df.head(args.max_points).copy()

    results = []
    failed = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        try:
            out = save_one_point(
                row=row,
                output_dir=args.output_dir,
                patch_size=args.patch_size_px,
                overview_margin_px=args.overview_margin_px,
            )
            results.append(out)
            if i % 10 == 0 or i == len(df):
                print(f"[INFO] processed {i}/{len(df)}")
        except Exception as e:
            failed.append({"point_id": row["point_id"], "error": str(e)})
            print(f"[WARN] failed point_id={row['point_id']} | {e}")

    pd.DataFrame(results).to_csv(os.path.join(args.output_dir, "rendered_files.csv"), index=False)
    pd.DataFrame(failed).to_csv(os.path.join(args.output_dir, "failed_points.csv"), index=False)

    print("\nDone")
    print("saved manifest:", os.path.join(args.output_dir, "rendered_files.csv"))
    print("saved failures:", os.path.join(args.output_dir, "failed_points.csv"))
    print("success:", len(results))
    print("failed :", len(failed))


if __name__ == "__main__":
    main()