import csv
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class InputPoint:
    point_id: str
    image_path: str
    x: float
    y: float
    target_label: Optional[str] = None


def load_points_csv(
    csv_path: str,
    tile_column: str = "image_path",
    point_id_column: str = "point_id",
    x_column: str = "x",
    y_column: str = "y",
    target_label_column: str = "target_label",
) -> List[InputPoint]:
    points = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            point_id = row.get(point_id_column, f"point_{i:06d}")
            image_path = row[tile_column]
            x = float(row[x_column])
            y = float(row[y_column])
            target_label = row.get(target_label_column, None)
            if target_label == "":
                target_label = None

            points.append(
                InputPoint(
                    point_id=point_id,
                    image_path=image_path,
                    x=x,
                    y=y,
                    target_label=target_label,
                )
            )
    return points