#!/usr/bin/env python3
"""
Collect MIL factorial experiment metrics into CSV and Markdown summaries.

This script intentionally uses only the Python standard library so it can run
in the login environment after SLURM jobs finish.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


FIELDNAMES = [
    "run_name",
    "status",
    "encoder",
    "train_image_mode",
    "eval_image_mode",
    "patch_size_px",
    "pooling",
    "bag_layout",
    "bag_instances",
    "conv_kernel_size",
    "seed",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "pos_precision",
    "pos_recall",
    "pos_f1",
    "pos_support",
    "average_precision",
    "roc_auc",
    "positive_prevalence",
    "positive_count",
    "negative_count",
    "best_threshold",
    "best_threshold_accuracy",
    "best_threshold_precision",
    "best_threshold_recall",
    "best_threshold_f1",
    "best_threshold_pred_positive",
    "all_positive_f1_baseline",
    "best_epoch",
    "best_val_macro_f1",
    "best_val_pos_f1",
    "best_val_average_precision",
    "best_val_roc_auc",
    "init_ckpt",
    "output_dir",
]

RANKING_COLUMNS = [
    "rank",
    "run_name",
    "encoder",
    "train_image_mode",
    "patch_size_px",
    "pooling",
    "best_threshold_f1",
    "average_precision",
    "roc_auc",
    "macro_f1",
    "pos_f1",
    "best_threshold",
]


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_key_values(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def to_int(value: Any) -> Optional[int]:
    number = to_float(value)
    if number is None:
        return None
    return int(round(number))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def best_threshold_row(path: Path) -> Dict[str, Any]:
    rows = read_csv_rows(path)
    if not rows:
        return {}

    def sort_key(row: Dict[str, str]) -> tuple:
        f1 = to_float(row.get("f1_shihuahuaco"))
        precision = to_float(row.get("precision_shihuahuaco"))
        recall = to_float(row.get("recall_shihuahuaco"))
        threshold = to_float(row.get("threshold"))
        return (
            -1.0 if f1 is None else f1,
            -1.0 if precision is None else precision,
            -1.0 if recall is None else recall,
            1.0 if threshold is None else -threshold,
        )

    return max(rows, key=sort_key)


def best_history_row(path: Path, monitor_col: str = "val_macro_f1") -> Dict[str, Any]:
    rows = read_csv_rows(path)
    if not rows:
        return {}

    def sort_key(row: Dict[str, str]) -> float:
        value = to_float(row.get(monitor_col))
        return -float("inf") if value is None else value

    return max(rows, key=sort_key)


def infer_from_run_name(run_name: str) -> Dict[str, str]:
    parts = run_name.split("_")
    out: Dict[str, str] = {}
    if parts:
        out["encoder"] = parts[0]
    if "patch" in run_name:
        for part in parts:
            if part.startswith("patch"):
                out["patch_size_px"] = part.replace("patch", "", 1)
                break
    for pooling in ("conv_lse", "conv_max", "conv_topk", "lse", "max", "topk"):
        if f"_{pooling}_" in f"_{run_name}_":
            out["pooling"] = pooling
            break
    return out


def collect_run(run_dir: Path) -> Dict[str, Any]:
    run_name = run_dir.name
    metadata = read_key_values(run_dir / "run_metadata.txt")
    config = load_json(run_dir / "mil_config.json")
    report = load_json(run_dir / "classification_report.json")
    diagnostics = load_json(run_dir / "binary_score_diagnostics.json")
    threshold = best_threshold_row(run_dir / "threshold_tuning.csv")
    history = best_history_row(run_dir / "training_history.csv")
    inferred = infer_from_run_name(run_name)

    pos = report.get("1", {}) if isinstance(report, dict) else {}
    macro = report.get("macro avg", {}) if isinstance(report, dict) else {}
    weighted = report.get("weighted avg", {}) if isinstance(report, dict) else {}

    complete = bool(report and diagnostics and threshold)

    row: Dict[str, Any] = {
        "run_name": metadata.get("run_name", run_name),
        "status": "complete" if complete else "incomplete",
        "encoder": metadata.get("encoder", inferred.get("encoder", "")),
        "train_image_mode": metadata.get("train_image_mode", config.get("image_mode", "")),
        "eval_image_mode": metadata.get("eval_image_mode", config.get("eval_image_mode", "")),
        "patch_size_px": to_int(metadata.get("patch_size_px", config.get("patch_size_px", inferred.get("patch_size_px")))),
        "pooling": metadata.get("pooling", config.get("pooling", inferred.get("pooling", ""))),
        "bag_layout": metadata.get("bag_layout", config.get("bag_layout", "")),
        "bag_instances": to_int(metadata.get("bag_instances", config.get("bag_instances"))),
        "conv_kernel_size": to_int(metadata.get("conv_kernel_size", config.get("conv_kernel_size"))),
        "seed": to_int(metadata.get("seed", config.get("seed"))),
        "accuracy": to_float(report.get("accuracy") if isinstance(report, dict) else None),
        "macro_f1": to_float(macro.get("f1-score")),
        "weighted_f1": to_float(weighted.get("f1-score")),
        "pos_precision": to_float(pos.get("precision")),
        "pos_recall": to_float(pos.get("recall")),
        "pos_f1": to_float(pos.get("f1-score")),
        "pos_support": to_int(pos.get("support")),
        "average_precision": to_float(diagnostics.get("average_precision")),
        "roc_auc": to_float(diagnostics.get("roc_auc")),
        "positive_prevalence": to_float(diagnostics.get("positive_prevalence")),
        "positive_count": to_int(diagnostics.get("positive_count")),
        "negative_count": to_int(diagnostics.get("negative_count")),
        "best_threshold": to_float(threshold.get("threshold")),
        "best_threshold_accuracy": to_float(threshold.get("accuracy")),
        "best_threshold_precision": to_float(threshold.get("precision_shihuahuaco")),
        "best_threshold_recall": to_float(threshold.get("recall_shihuahuaco")),
        "best_threshold_f1": to_float(threshold.get("f1_shihuahuaco")),
        "best_threshold_pred_positive": to_int(threshold.get("pred_positive")),
        "all_positive_f1_baseline": to_float(threshold.get("f1_all_positive_baseline")),
        "best_epoch": to_int(history.get("epoch")),
        "best_val_macro_f1": to_float(history.get("val_macro_f1")),
        "best_val_pos_f1": to_float(history.get("val_pos_f1")),
        "best_val_average_precision": to_float(history.get("val_average_precision")),
        "best_val_roc_auc": to_float(history.get("val_roc_auc")),
        "init_ckpt": metadata.get("init_ckpt", config.get("init_ckpt", "")),
        "output_dir": str(run_dir),
    }
    return row


def sort_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def sort_key(row: Dict[str, Any]) -> tuple:
        f1 = row.get("best_threshold_f1")
        ap = row.get("average_precision")
        roc = row.get("roc_auc")
        return (
            row.get("status") == "complete",
            -1.0 if f1 is None else float(f1),
            -1.0 if ap is None else float(ap),
            -1.0 if roc is None else float(roc),
        )

    return sorted(rows, key=sort_key, reverse=True)


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_summary_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def write_summary_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    complete = [row for row in rows if row.get("status") == "complete"]
    incomplete = [row for row in rows if row.get("status") != "complete"]

    lines: List[str] = []
    lines.append("# MIL Shihuahuaco Factorial Summary")
    lines.append("")
    lines.append(f"Complete runs: {len(complete)}")
    lines.append(f"Incomplete runs: {len(incomplete)}")
    lines.append("")

    if complete:
        ranked = []
        for idx, row in enumerate(complete, start=1):
            ranked_row = dict(row)
            ranked_row["rank"] = idx
            ranked.append(ranked_row)

        lines.append("## Top Runs")
        lines.append("")
        lines.append("| " + " | ".join(RANKING_COLUMNS) + " |")
        lines.append("| " + " | ".join(["---"] * len(RANKING_COLUMNS)) + " |")
        for row in ranked:
            lines.append("| " + " | ".join(format_cell(row.get(col)) for col in RANKING_COLUMNS) + " |")
        lines.append("")
    else:
        lines.append("No complete runs found yet.")
        lines.append("")

    if incomplete:
        lines.append("## Incomplete Runs")
        lines.append("")
        lines.append("| run_name | output_dir |")
        lines.append("| --- | --- |")
        for row in incomplete:
            lines.append(f"| {row.get('run_name', '')} | {row.get('output_dir', '')} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect MIL factorial results.")
    default_root = Path(__file__).resolve().parent
    scratch_root = Path("/mnt/parscratch/users/aca21jo/realign_experiments/mil_shihuaco_factorial")
    default_results = scratch_root if scratch_root.exists() else default_root / "results"
    parser.add_argument("--results_root", default=str(default_results))
    parser.add_argument("--output_csv", default=str(default_root / "summary.csv"))
    parser.add_argument("--output_md", default=str(default_root / "summary.md"))
    args = parser.parse_args()

    results_root = Path(args.results_root)
    run_dirs = sorted(path for path in results_root.iterdir() if path.is_dir()) if results_root.exists() else []
    rows = sort_rows(collect_run(path) for path in run_dirs)

    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    write_summary_csv(output_csv, rows)
    write_summary_md(output_md, rows)

    print(f"Runs scanned : {len(run_dirs)}")
    print(f"Summary CSV  : {output_csv}")
    print(f"Summary MD   : {output_md}")


if __name__ == "__main__":
    main()
