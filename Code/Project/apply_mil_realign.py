"""
Apply a trained MIL model to an arbitrary point file and save realigned points.

The input file must contain the fields used by the MIL point-bag dataset:
`Folder`, `File`, `fx`, `fy`, and a binary label field. For weak census
Shihuahuaco points, set `BinaryTree=1` so the positive search radius is used.
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.checkpoint import load_encoder_from_checkpoint
from train_mil_classifier import (
    Config,
    MILPointBagDataset,
    build_mil_head,
    forward_bags,
)
from train_supervised_encoder import (
    build_eval_transform,
    build_tif_index,
    infer_feature_dim,
    safe_mkdir,
)


def load_config(run_dir: Path) -> Config:
    cfg_path = run_dir / "mil_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        return Config(**json.load(f))


def load_head(head_path: Path, feat_dim: int, device, cfg: Config):
    state = torch.load(head_path, map_location=device)
    head = build_mil_head(feat_dim, cfg, device)
    head.load_state_dict(state["head_state_dict"])
    head.eval()
    return head


def best_threshold(run_dir: Path, fallback: float):
    path = run_dir / "threshold_tuning.csv"
    if not path.exists():
        return float(fallback)
    df = pd.read_csv(path)
    if "f1_shihuahuaco" not in df.columns or "threshold" not in df.columns or df.empty:
        return float(fallback)
    row = df.sort_values("f1_shihuahuaco", ascending=False).iloc[0]
    return float(row["threshold"])


def pixel_to_world_cache():
    cache = {}

    def convert(image_path, px, py):
        if image_path not in cache:
            src = rasterio.open(image_path)
            cache[image_path] = src
        src = cache[image_path]
        x, y = src.xy(float(py), float(px))
        return float(x), float(y)

    def close():
        for src in cache.values():
            src.close()
        cache.clear()

    return convert, close


def parse_args():
    p = argparse.ArgumentParser(description="Apply trained MIL model to point file and save realigned coordinates.")
    p.add_argument("--mil_output_dir", required=True)
    p.add_argument("--input_points", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--output_gpkg", default="")
    p.add_argument("--checkpoint_name", default="best", choices=["best", "last"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--decision_threshold", type=float, default=-1.0)
    p.add_argument("--selection", default="raw", choices=["raw", "context"])
    p.add_argument("--coord_mode", default="", choices=["", "auto", "normalized", "pixel", "world"])
    p.add_argument("--label_field", default="")
    p.add_argument("--folder_field", default="")
    p.add_argument("--file_field", default="")
    p.add_argument("--fx_field", default="")
    p.add_argument("--fy_field", default="")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    run_dir = Path(args.mil_output_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_gpkg = Path(args.output_gpkg) if args.output_gpkg else None
    if output_gpkg:
        output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_config(run_dir)
    cfg.val_shp = str(args.input_points)
    cfg.output_dir = str(output_csv.parent)
    if args.coord_mode:
        cfg.coord_mode = args.coord_mode
    if args.label_field:
        cfg.label_field = args.label_field
    if args.folder_field:
        cfg.folder_field = args.folder_field
    if args.file_field:
        cfg.file_field = args.file_field
    if args.fx_field:
        cfg.fx_field = args.fx_field
    if args.fy_field:
        cfg.fy_field = args.fy_field

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    encoder_path = run_dir / f"phase1_encoder_{args.checkpoint_name}.pth"
    head_path = run_dir / f"mil_head_{args.checkpoint_name}.pth"
    if not encoder_path.exists():
        raise FileNotFoundError(encoder_path)
    if not head_path.exists():
        raise FileNotFoundError(head_path)

    model, _ = load_encoder_from_checkpoint(str(encoder_path), device)
    feat_dim = infer_feature_dim(model, device, cfg.image_size)
    head = load_head(head_path, feat_dim, device, cfg)

    folder_to_paths = build_tif_index(cfg.imagery_root)
    dataset = MILPointBagDataset(
        shp_path=str(args.input_points),
        imagery_root=cfg.imagery_root,
        label_field=cfg.label_field,
        folder_field=cfg.folder_field,
        file_field=cfg.file_field,
        fx_field=cfg.fx_field,
        fy_field=cfg.fy_field,
        coord_mode=cfg.coord_mode,
        positive_class=cfg.positive_class,
        patch_size_px=cfg.patch_size_px,
        transform=build_eval_transform(cfg.image_size, cfg.eval_image_mode),
        bag_radius_m=cfg.bag_radius_m,
        negative_bag_radius_m=cfg.negative_bag_radius_m,
        bag_instances=cfg.bag_instances,
        bag_layout=cfg.bag_layout,
        max_black_fraction=cfg.max_black_fraction,
        max_bright_fraction=cfg.max_bright_fraction,
        folder_to_paths=folder_to_paths,
        repeat_factor=1,
        train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    threshold = best_threshold(run_dir, 0.5) if args.decision_threshold < 0 else float(args.decision_threshold)
    to_world, close_world = pixel_to_world_cache()

    rows = []
    try:
        for bags, y, idxs, offsets in tqdm(loader, desc="MIL realign", dynamic_ncols=True):
            bags = bags.to(device, non_blocking=True)
            bag_logits, context_logits, raw_logits = forward_bags(
                model,
                head,
                bags,
                cfg.pooling,
                cfg.lse_tau,
                cfg.topk,
            )
            bag_probs = torch.sigmoid(bag_logits).detach().cpu()
            context_probs = torch.sigmoid(context_logits).detach().cpu()
            raw_probs = torch.sigmoid(raw_logits).detach().cpu()
            raw_best = raw_logits.argmax(dim=1).detach().cpu()
            context_best = context_logits.argmax(dim=1).detach().cpu()
            idxs_cpu = idxs.detach().cpu()
            offsets_cpu = offsets.detach().cpu()
            y_cpu = y.detach().cpu()

            for j in range(len(idxs_cpu)):
                sample = dataset.samples[int(idxs_cpu[j])]
                input_row = dataset.gdf.iloc[int(sample["source_index"])].drop(labels="geometry", errors="ignore").to_dict()

                raw_i = int(raw_best[j])
                context_i = int(context_best[j])
                selected_i = raw_i if args.selection == "raw" else context_i

                raw_dx_px = float(offsets_cpu[j, raw_i, 0])
                raw_dy_px = float(offsets_cpu[j, raw_i, 1])
                context_dx_px = float(offsets_cpu[j, context_i, 0])
                context_dy_px = float(offsets_cpu[j, context_i, 1])
                selected_dx_px = float(offsets_cpu[j, selected_i, 0])
                selected_dy_px = float(offsets_cpu[j, selected_i, 1])

                raw_px = sample["center_px"] + raw_dx_px
                raw_py = sample["center_py"] + raw_dy_px
                context_px = sample["center_px"] + context_dx_px
                context_py = sample["center_py"] + context_dy_px
                selected_px = sample["center_px"] + selected_dx_px
                selected_py = sample["center_py"] + selected_dy_px

                raw_x, raw_y = to_world(sample["image_path"], raw_px, raw_py)
                context_x, context_y = to_world(sample["image_path"], context_px, context_py)
                selected_x, selected_y = to_world(sample["image_path"], selected_px, selected_py)

                bag_prob = float(bag_probs[j])
                out = dict(input_row)
                out.update(
                    {
                        "model_run": run_dir.name,
                        "checkpoint_name": args.checkpoint_name,
                        "selection": args.selection,
                        "decision_threshold": threshold,
                        "bag_prob_1": bag_prob,
                        "is_positive_at_threshold": int(bag_prob >= threshold),
                        "weak_y_true": int(float(y_cpu[j])),
                        "original_x": float(sample["raw_x"]),
                        "original_y": float(sample["raw_y"]),
                        "center_px": float(sample["center_px"]),
                        "center_py": float(sample["center_py"]),
                        "pixel_size_x": float(sample["pixel_size_x"]),
                        "pixel_size_y": float(sample["pixel_size_y"]),
                        "coord_mode_used": sample["coord_mode_used"],
                        "image_path": sample["image_path"],
                        "raw_selected_instance": raw_i,
                        "raw_selected_prob_1": float(raw_probs[j, raw_i]),
                        "raw_context_prob_1_at_raw_selection": float(context_probs[j, raw_i]),
                        "raw_dx_px": raw_dx_px,
                        "raw_dy_px": raw_dy_px,
                        "raw_dx_m": raw_dx_px * sample["pixel_size_x"],
                        "raw_dy_m": raw_dy_px * sample["pixel_size_y"],
                        "raw_offset_m": float(((raw_dx_px * sample["pixel_size_x"]) ** 2 + (raw_dy_px * sample["pixel_size_y"]) ** 2) ** 0.5),
                        "raw_realigned_x": raw_x,
                        "raw_realigned_y": raw_y,
                        "context_selected_instance": context_i,
                        "context_selected_prob_1": float(context_probs[j, context_i]),
                        "context_raw_prob_1_at_context_selection": float(raw_probs[j, context_i]),
                        "context_dx_px": context_dx_px,
                        "context_dy_px": context_dy_px,
                        "context_dx_m": context_dx_px * sample["pixel_size_x"],
                        "context_dy_m": context_dy_px * sample["pixel_size_y"],
                        "context_offset_m": float(((context_dx_px * sample["pixel_size_x"]) ** 2 + (context_dy_px * sample["pixel_size_y"]) ** 2) ** 0.5),
                        "context_realigned_x": context_x,
                        "context_realigned_y": context_y,
                        "selection_disagrees_with_context": int(raw_i != context_i),
                        "selected_instance": selected_i,
                        "selected_dx_px": selected_dx_px,
                        "selected_dy_px": selected_dy_px,
                        "selected_dx_m": selected_dx_px * sample["pixel_size_x"],
                        "selected_dy_m": selected_dy_px * sample["pixel_size_y"],
                        "selected_offset_m": float(((selected_dx_px * sample["pixel_size_x"]) ** 2 + (selected_dy_px * sample["pixel_size_y"]) ** 2) ** 0.5),
                        "realigned_x": selected_x,
                        "realigned_y": selected_y,
                        "realigned_delta_x": selected_x - float(sample["raw_x"]),
                        "realigned_delta_y": selected_y - float(sample["raw_y"]),
                    }
                )
                rows.append(out)
    finally:
        close_world()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    summary = {
        "mil_output_dir": str(run_dir),
        "input_points": str(args.input_points),
        "checkpoint_name": args.checkpoint_name,
        "selection": args.selection,
        "decision_threshold": threshold,
        "rows": int(len(df)),
        "mean_bag_prob_1": float(df["bag_prob_1"].mean()) if len(df) else None,
        "pred_positive_at_threshold": int(df["is_positive_at_threshold"].sum()) if len(df) else 0,
        "mean_selected_offset_m": float(df["selected_offset_m"].mean()) if len(df) else None,
        "output_csv": str(output_csv),
        "output_gpkg": str(output_gpkg) if output_gpkg else "",
    }
    with (output_csv.parent / "realign_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if output_gpkg:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["realigned_x"], df["realigned_y"]),
            crs="EPSG:32718",
        )
        gdf.to_file(output_gpkg, driver="GPKG")

    print(json.dumps(summary, indent=2))
    print(f"CSV : {output_csv}")
    if output_gpkg:
        print(f"GPKG: {output_gpkg}")


if __name__ == "__main__":
    main()

