import argparse
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import torch
import torch.nn as nn
from PIL import Image
import rasterio
from torchvision import transforms
from tqdm import tqdm

from src.models.checkpoint import load_encoder_from_checkpoint


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_shp", required=True)
    p.add_argument("--encoder_ckpt", required=True)
    p.add_argument("--head_ckpt", required=True)
    p.add_argument("--imagery_root", required=True)

    p.add_argument("--tile_column", default="matched_tif")
    p.add_argument("--label_column", default="label")
    p.add_argument("--target_label", default="Shihuahuaco")
    p.add_argument("--x_column", default="gt_east")
    p.add_argument("--y_column", default="gt_north")
    p.add_argument("--crs", default="EPSG:32718")

    p.add_argument("--grid_sizes", default="30,20,10")
    p.add_argument("--threshold", type=float, default=0.18)
    p.add_argument("--min_realigned_boxes", type=int, default=3)
    p.add_argument("--max_iterations", type=int, default=10)

    p.add_argument("--positive_class", default="1")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--patch_size_px", type=int, default=224)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_amp", action="store_true")

    return p.parse_args()


def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def forward_features(model, x):
    if hasattr(model, "encode"):
        z = model.encode(x)
    else:
        z = model(x)

    if isinstance(z, dict):
        for k in ["features", "embedding", "embeddings", "x", "last_hidden_state"]:
            if k in z:
                z = z[k]
                break

    if isinstance(z, (tuple, list)):
        z = z[0]

    if z.ndim == 4:
        z = z.mean(dim=(2, 3))
    elif z.ndim == 3:
        z = z[:, 0]

    return z


def infer_feature_dim(model, device, image_size):
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 3, image_size, image_size).to(device)
        z = forward_features(model, x)
    return int(z.shape[1])


def read_patch(image_path, x, y, size):
    half = size // 2

    with rasterio.open(image_path) as src:
        window = rasterio.windows.Window(
            int(round(x)) - half,
            int(round(y)) - half,
            size,
            size,
        )
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


def world_to_pixel(path, east, north):
    with rasterio.open(path) as src:
        row, col = src.index(float(east), float(north))
    return float(col), float(row)


def resolve_tile(imagery_root, value):
    value = str(value)

    if os.path.exists(value):
        return value

    base = os.path.basename(value)

    for root, _, files in os.walk(imagery_root):
        for f in files:
            if f == base:
                return os.path.join(root, f)

    raise FileNotFoundError(f"Could not resolve tile: {value}")


@torch.no_grad()
def get_prob(model, head, transform, image_path, east, north, target_idx, patch_size, device, use_amp):
    px, py = world_to_pixel(image_path, east, north)
    patch = read_patch(image_path, px, py, patch_size)
    x = transform(patch).unsqueeze(0).to(device)

    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
        z = forward_features(model, x)
        logits = head(z)
        prob = torch.softmax(logits, dim=1)[0, target_idx]

    return float(prob.detach().cpu())


def create_3x3_grid(center_point, cell_size):
    x, y = center_point.x, center_point.y
    boxes = []

    for i in range(-1, 2):
        for j in range(-1, 2):
            minx = x + j * cell_size - cell_size / 2
            maxx = x + j * cell_size + cell_size / 2
            miny = y - i * cell_size - cell_size / 2
            maxy = y - i * cell_size + cell_size / 2

            boxes.append(
                Polygon([
                    (minx, miny),
                    (maxx, miny),
                    (maxx, maxy),
                    (minx, maxy),
                ])
            )

    return boxes


def process_one_stage(
    points,
    model,
    head,
    transform,
    target_idx,
    grid_sizes,
    threshold,
    min_realigned_boxes,
    patch_size,
    device,
    use_amp,
):
    new_centers = []
    new_slides = []

    for _, row in tqdm(points.iterrows(), total=len(points), dynamic_ncols=True):
        point = row.geometry
        image_path = row["resolved_tif"]

        row_dict = row.drop(labels="geometry").to_dict()

        stage_idx = int(row_dict.get("stage_idx", 0))
        stage_idx = min(stage_idx, len(grid_sizes) - 1)
        cell_size = grid_sizes[stage_idx]

        boxes = create_3x3_grid(point, cell_size)

        scored = []
        all_probs = []

        for cell_id, box in enumerate(boxes):
            prob = get_prob(
                model=model,
                head=head,
                transform=transform,
                image_path=image_path,
                east=box.centroid.x,
                north=box.centroid.y,
                target_idx=target_idx,
                patch_size=patch_size,
                device=device,
                use_amp=use_amp,
            )
            all_probs.append(prob)

            if prob >= threshold:
                scored.append((box, prob, cell_id))

        detected_count = len(scored)
        max_prob = float(np.max(all_probs))
        mean_prob = float(np.mean(all_probs))

        row_dict[f"stage_{stage_idx}_grid_size_m"] = cell_size
        row_dict[f"stage_{stage_idx}_max_prob"] = max_prob
        row_dict[f"stage_{stage_idx}_mean_prob"] = mean_prob
        row_dict[f"stage_{stage_idx}_detected_boxes"] = detected_count

        row_dict["last_stage_idx"] = stage_idx
        row_dict["last_grid_size_m"] = cell_size
        row_dict["last_max_prob"] = max_prob
        row_dict["last_mean_prob"] = mean_prob
        row_dict["last_detected_boxes"] = detected_count

        if detected_count == 0:
            row_dict["status"] = "research"
            row_dict["final_reason"] = "no_detected_boxes"
            new_centers.append({**row_dict, "geometry": point})
            continue

        avg_x = float(np.mean([b.centroid.x for b, _, _ in scored]))
        avg_y = float(np.mean([b.centroid.y for b, _, _ in scored]))
        new_point = Point(avg_x, avg_y)

        # If enough boxes detected, finalize immediately as realigned.
        if detected_count >= min_realigned_boxes:
            row_dict["status"] = "realigned"
            row_dict["final_reason"] = f"{detected_count}_boxes_above_threshold"
            new_centers.append({**row_dict, "geometry": new_point})
            continue

        # If 1-2 boxes detected and not at final stage, slide to next stage.
        if stage_idx < len(grid_sizes) - 1:
            row_dict["status"] = "slide"
            row_dict["final_reason"] = f"{detected_count}_boxes_slide_to_next_stage"
            row_dict["stage_idx"] = stage_idx + 1
            new_slides.append({**row_dict, "geometry": new_point})
            continue

        # At smallest grid, 1-2 detections: finalize as slide_final.
        row_dict["status"] = "slide_final"
        row_dict["final_reason"] = f"{detected_count}_boxes_at_final_stage"
        new_centers.append({**row_dict, "geometry": new_point})

    crs = points.crs

    centers = (
        gpd.GeoDataFrame(new_centers, crs=crs)
        if new_centers
        else gpd.GeoDataFrame(columns=list(points.columns), geometry="geometry", crs=crs)
    )

    slides = (
        gpd.GeoDataFrame(new_slides, crs=crs)
        if new_slides
        else gpd.GeoDataFrame(columns=list(points.columns), geometry="geometry", crs=crs)
    )

    return centers, slides


def main():
    args = parse_args()

    grid_sizes = [float(x) for x in args.grid_sizes.split(",")]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    os.makedirs(os.path.dirname(args.output_shp), exist_ok=True)

    model, _ = load_encoder_from_checkpoint(args.encoder_ckpt, device)
    model.eval()

    head_ckpt = torch.load(args.head_ckpt, map_location=device)
    classes = [str(c) for c in head_ckpt["classes"]]
    class_to_idx = {str(k): int(v) for k, v in head_ckpt["class_to_idx"].items()}

    feat_dim = infer_feature_dim(model, device, args.image_size)
    head = nn.Linear(feat_dim, len(classes)).to(device)
    head.load_state_dict(head_ckpt["head_state_dict"])
    head.eval()

    if str(args.positive_class) not in class_to_idx:
        raise ValueError(f"positive_class={args.positive_class} not in class_to_idx={class_to_idx}")

    target_idx = class_to_idx[str(args.positive_class)]
    transform = build_transform(args.image_size)

    df = pd.read_csv(args.input_csv)
    df[args.label_column] = df[args.label_column].astype(str).str.strip()
    df = df[df[args.label_column] == args.target_label].copy()

    if len(df) == 0:
        raise ValueError(f"No rows found for target_label={args.target_label}")

    df["resolved_tif"] = df[args.tile_column].apply(lambda x: resolve_tile(args.imagery_root, x))
    df["start_east"] = df[args.x_column].astype(float)
    df["start_north"] = df[args.y_column].astype(float)
    df["stage_idx"] = 0

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[args.x_column], df[args.y_column]),
        crs=args.crs,
    )

    current = gdf
    final = []

    print("=" * 80)
    print("Slide-grid classifier Phase 3")
    print("Input:", args.input_csv)
    print("Output:", args.output_shp)
    print("Target:", args.target_label)
    print("Rows:", len(gdf))
    print("Grid sizes:", grid_sizes)
    print("Threshold:", args.threshold)
    print("Min realigned boxes:", args.min_realigned_boxes)
    print("Max iterations:", args.max_iterations)
    print("Device:", device)
    print("=" * 80)

    for iteration in range(1, args.max_iterations + 1):
        if current.empty:
            print("All points finalized.")
            break

        print(f"\nIteration {iteration}/{args.max_iterations} | active slides: {len(current)}")

        centers, slides = process_one_stage(
            points=current,
            model=model,
            head=head,
            transform=transform,
            target_idx=target_idx,
            grid_sizes=grid_sizes,
            threshold=args.threshold,
            min_realigned_boxes=args.min_realigned_boxes,
            patch_size=args.patch_size_px,
            device=device,
            use_amp=use_amp,
        )

        if not centers.empty:
            centers["final_iteration"] = iteration
            final.append(centers)

        current = slides

        print(f"  finalized: {len(centers)} | continuing slides: {len(slides)}")

    if not current.empty:
        current["status"] = "forced_final"
        current["final_reason"] = "max_iterations_reached"
        current["final_iteration"] = args.max_iterations
        final.append(current)

    final_gdf = pd.concat(final, ignore_index=True)
    final_gdf = gpd.GeoDataFrame(final_gdf, geometry="geometry", crs=args.crs)

    final_gdf["final_east"] = final_gdf.geometry.x
    final_gdf["final_north"] = final_gdf.geometry.y

    if "gt_east" in final_gdf.columns and "gt_north" in final_gdf.columns:
        final_gdf["dist_before_m"] = np.sqrt(
            (final_gdf["start_east"] - final_gdf["gt_east"]) ** 2
            + (final_gdf["start_north"] - final_gdf["gt_north"]) ** 2
        )
        final_gdf["dist_after_m"] = np.sqrt(
            (final_gdf["final_east"] - final_gdf["gt_east"]) ** 2
            + (final_gdf["final_north"] - final_gdf["gt_north"]) ** 2
        )
        final_gdf["improvement_m"] = final_gdf["dist_before_m"] - final_gdf["dist_after_m"]

    final_gdf.to_file(args.output_shp)

    output_csv = os.path.splitext(args.output_shp)[0] + ".csv"
    final_gdf.drop(columns="geometry").to_csv(output_csv, index=False)

    print("=" * 80)
    print("Done")
    print("Saved SHP:", args.output_shp)
    print("Saved CSV:", output_csv)
    print("=" * 80)


if __name__ == "__main__":
    main()