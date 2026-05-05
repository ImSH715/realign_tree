import argparse, json, os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from train_supervised_encoder import (
    GTPointDataset, build_eval_transform, build_tif_index,
    forward_features, infer_feature_dim
)
from src.models.checkpoint import load_encoder_from_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_ckpt", required=True)
    p.add_argument("--head_ckpt", required=True)
    p.add_argument("--gt_path", required=True)
    p.add_argument("--imagery_root", required=True)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--label_field", default="BinaryTree")
    p.add_argument("--folder_field", default="Folder")
    p.add_argument("--file_field", default="File")
    p.add_argument("--fx_field", default="fx")
    p.add_argument("--fy_field", default="fy")
    p.add_argument("--coord_mode", default="auto")

    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--patch_size_px", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--image_mode", default="rgb", choices=["rgb", "grayscale"])
    return p.parse_args()


def binary_score_diagnostics(df, classes):
    if len(classes) != 2:
        return {}

    class_names = [str(c) for c in classes]
    positive_name = "1" if "1" in class_names else class_names[-1]
    prob_col = f"prob_{positive_name}"
    if prob_col not in df.columns:
        return {}

    pos_idx = class_names.index(positive_name)
    y_true = (df["y_true"].astype(int).values == pos_idx).astype(int)
    y_score = df[prob_col].astype(float).values

    diagnostics = {
        "positive_class": positive_name,
        "positive_count": int(y_true.sum()),
        "negative_count": int((1 - y_true).sum()),
        "positive_prevalence": float(y_true.mean()) if len(y_true) else 0.0,
        "probability_column": prob_col,
    }

    if len(np.unique(y_true)) == 2:
        diagnostics["average_precision"] = float(average_precision_score(y_true, y_score))
        diagnostics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        diagnostics["average_precision"] = None
        diagnostics["roc_auc"] = None

    grouped = df.assign(_binary_true=y_true).groupby("_binary_true")[prob_col].describe()
    grouped.index = grouped.index.map({0: "negative", 1: "positive"})
    diagnostics["score_summary_by_true_class"] = json.loads(grouped.to_json(orient="index"))

    if y_true.sum() and (1 - y_true).sum():
        pos_mean = float(y_score[y_true == 1].mean())
        neg_mean = float(y_score[y_true == 0].mean())
        diagnostics["positive_mean_minus_negative_mean"] = pos_mean - neg_mean

    return diagnostics


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, _ = load_encoder_from_checkpoint(args.encoder_ckpt, device)

    head_ckpt = torch.load(args.head_ckpt, map_location=device)
    classes = head_ckpt["classes"]
    class_to_idx = head_ckpt["class_to_idx"]

    feat_dim = infer_feature_dim(model, device, args.image_size)
    head = nn.Linear(feat_dim, len(classes)).to(device)
    head.load_state_dict(head_ckpt["head_state_dict"])

    folder_to_paths = build_tif_index(args.imagery_root)

    ds = GTPointDataset(
        shp_path=args.gt_path,
        imagery_root=args.imagery_root,
        label_field=args.label_field,
        folder_field=args.folder_field,
        file_field=args.file_field,
        fx_field=args.fx_field,
        fy_field=args.fy_field,
        patch_size_px=args.patch_size_px,
        transform=build_eval_transform(args.image_size, args.image_mode),
        coord_mode=args.coord_mode,
        class_to_idx=class_to_idx,
        folder_to_paths=folder_to_paths,
        debug_patches=0,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    head.eval()

    y_true, y_pred = [], []
    probs_all = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        z = forward_features(model, x)
        logits = head(z)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        probs_all.extend(probs.cpu().numpy().tolist())

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=[str(c) for c in classes],
        zero_division=0,
        output_dict=True,
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    with open(os.path.join(args.output_dir, "classifier_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        os.path.join(args.output_dir, "classifier_confusion_matrix.csv")
    )

    out = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    })

    for i, cls in enumerate(classes):
        out[f"prob_{cls}"] = [p[i] for p in probs_all]

    out.to_csv(os.path.join(args.output_dir, "classifier_predictions.csv"), index=False)

    diagnostics = binary_score_diagnostics(out, classes)
    if diagnostics:
        with open(os.path.join(args.output_dir, "binary_score_diagnostics.json"), "w") as f:
            json.dump(diagnostics, f, indent=2)
        pd.DataFrame(diagnostics["score_summary_by_true_class"]).T.to_csv(
            os.path.join(args.output_dir, "binary_score_summary_by_true_class.csv")
        )
        print("Binary score diagnostics:")
        print(json.dumps(diagnostics, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
