import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from PIL import Image
import matplotlib.pyplot as plt

# =========================================================
# SETTINGS
# =========================================================
CSV_PATH = "./outputs/evaluation/single_tif_beta0002b_refined.csv"
POINT_ID = "shihuahuaco_top1_0000"
OUTPUT_DIR = "./outputs/evaluation/point_visualization"

PATCH_SIZE = 224  
OVERVIEW_MARGIN_PX = 350 

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# HELPERS
# =========================================================
def normalize_to_uint8(arr):
    arr = arr.astype(np.float32)

    if arr.ndim == 3 and arr.shape[0] in [1, 3, 4]:
        # bands, h, w -> h, w, bands
        arr = np.transpose(arr[:3], (1, 2, 0))
    elif arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]

    lo = np.nanpercentile(arr, 1)
    hi = np.nanpercentile(arr, 99)
    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
    arr = (arr * 255).astype(np.uint8)
    return arr

def read_window_rgb(src, col_center, row_center, width, height):
    left = int(round(col_center - width // 2))
    top = int(round(row_center - height // 2))
    window = Window(left, top, width, height)
    data = src.read(window=window, boundless=True, fill_value=0)

    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    elif data.shape[0] >= 3:
        data = data[:3]
    else:
        raise ValueError(f"Unexpected band count: {data.shape}")

    rgb = normalize_to_uint8(data)
    return rgb

def read_bbox_rgb(src, col_min, row_min, col_max, row_max):
    width = int(col_max - col_min)
    height = int(row_max - row_min)
    window = Window(int(col_min), int(row_min), width, height)
    data = src.read(window=window, boundless=True, fill_value=0)

    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    elif data.shape[0] >= 3:
        data = data[:3]
    else:
        raise ValueError(f"Unexpected band count: {data.shape}")

    rgb = normalize_to_uint8(data)
    return rgb

# =========================================================
# LOAD CSV AND PICK POINT
# =========================================================
df = pd.read_csv(CSV_PATH).copy()
df["point_id"] = df["point_id"].astype(str)

row = df[df["point_id"] == str(POINT_ID)]
if len(row) == 0:
    print("Available point_id examples:")
    print(df["point_id"].head(20).tolist())
    raise ValueError(f"point_id={POINT_ID} not found in {CSV_PATH}")

row = row.iloc[0]

image_path = row["image_path"] if "image_path" in row else row["matched_tif"]

orig_e = float(row["original_east"])
orig_n = float(row["original_north"])
ref_e = float(row["refined_east"])
ref_n = float(row["refined_north"])

has_gt = ("gt_east" in row.index) and ("gt_north" in row.index)
if has_gt:
    gt_e = float(row["gt_east"])
    gt_n = float(row["gt_north"])

print("Selected point:")
print("point_id   :", row["point_id"])
print("label      :", row["label"] if "label" in row.index else row["target_label"])
print("image_path :", image_path)
print("original   :", orig_e, orig_n)
print("refined    :", ref_e, ref_n)
if has_gt:
    print("gt         :", gt_e, gt_n)

# =========================================================
# READ TIFF AND EXTRACT PATCHES
# =========================================================
with rasterio.open(image_path) as src:
    orig_row, orig_col = src.index(orig_e, orig_n)
    ref_row, ref_col = src.index(ref_e, ref_n)

    if has_gt:
        gt_row, gt_col = src.index(gt_e, gt_n)

    original_patch = read_window_rgb(src, orig_col, orig_row, PATCH_SIZE, PATCH_SIZE)
    refined_patch = read_window_rgb(src, ref_col, ref_row, PATCH_SIZE, PATCH_SIZE)

    # overview crop bbox
    cols = [orig_col, ref_col]
    rows = [orig_row, ref_row]

    if has_gt:
        cols.append(gt_col)
        rows.append(gt_row)

    col_min = min(cols) - OVERVIEW_MARGIN_PX
    col_max = max(cols) + OVERVIEW_MARGIN_PX
    row_min = min(rows) - OVERVIEW_MARGIN_PX
    row_max = max(rows) + OVERVIEW_MARGIN_PX

    overview_rgb = read_bbox_rgb(src, col_min, row_min, col_max, row_max)

    # local coords inside overview crop
    orig_x_local = orig_col - col_min
    orig_y_local = orig_row - row_min

    ref_x_local = ref_col - col_min
    ref_y_local = ref_row - row_min

    if has_gt:
        gt_x_local = gt_col - col_min
        gt_y_local = gt_row - row_min

# =========================================================
# SAVE INDIVIDUAL PATCHES
# =========================================================
orig_patch_path = os.path.join(OUTPUT_DIR, f"point_{POINT_ID}_original_patch.png")
ref_patch_path = os.path.join(OUTPUT_DIR, f"point_{POINT_ID}_refined_patch.png")
overview_path = os.path.join(OUTPUT_DIR, f"point_{POINT_ID}_overview.png")
combined_path = os.path.join(OUTPUT_DIR, f"point_{POINT_ID}_combined.png")

Image.fromarray(original_patch).save(orig_patch_path)
Image.fromarray(refined_patch).save(ref_patch_path)

# =========================================================
# SAVE OVERVIEW
# =========================================================
plt.figure(figsize=(8, 8))
plt.imshow(overview_rgb)
plt.scatter(orig_x_local, orig_y_local, s=120, marker='o', label='original')
plt.scatter(ref_x_local, ref_y_local, s=120, marker='x', label='refined')

if has_gt:
    plt.scatter(gt_x_local, gt_y_local, s=120, marker='^', label='gt')

plt.legend()
plt.title(f"Overview | point_id={POINT_ID}")
plt.axis("off")
plt.tight_layout()
plt.savefig(overview_path, dpi=200)
plt.close()

# =========================================================
# SAVE COMBINED FIGURE
# =========================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# original patch
axes[0].imshow(original_patch)
axes[0].scatter(PATCH_SIZE // 2, PATCH_SIZE // 2, s=80, marker='o')
axes[0].set_title("Original point patch")
axes[0].axis("off")

# refined patch
axes[1].imshow(refined_patch)
axes[1].scatter(PATCH_SIZE // 2, PATCH_SIZE // 2, s=80, marker='x')
axes[1].set_title("Refined point patch")
axes[1].axis("off")

# overview
axes[2].imshow(overview_rgb)
axes[2].scatter(orig_x_local, orig_y_local, s=120, marker='o', label='original')
axes[2].scatter(ref_x_local, ref_y_local, s=120, marker='x', label='refined')
if has_gt:
    axes[2].scatter(gt_x_local, gt_y_local, s=120, marker='^', label='gt')
axes[2].legend()
axes[2].set_title("Overview")
axes[2].axis("off")

fig.suptitle(f"point_id={POINT_ID}", fontsize=14)
plt.tight_layout()
plt.savefig(combined_path, dpi=200)
plt.close()

print("\nSaved files:")
print(orig_patch_path)
print(ref_patch_path)
print(overview_path)
print(combined_path)