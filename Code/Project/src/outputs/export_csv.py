import os
import csv
from dataclasses import asdict


def save_refinement_results_csv(results, output_csv: str) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    if len(results) == 0:
        raise RuntimeError("No refinement results to save.")

    fieldnames = list(asdict(results[0]).keys())

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))