import argparse
import os
import pandas as pd
import rasterio


def pixel_to_world(image_path, px, py):
    with rasterio.open(image_path) as src:
        east, north = src.xy(float(py), float(px))
    return float(east), float(north)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--target_label", default="Shihuahuaco")
    p.add_argument(
        "--use_selected_point",
        action="store_true",
        help="Use MIL selected point (px, py). Otherwise use original center (center_px, center_py).",
    )
    args = p.parse_args()

    df = pd.read_csv(args.input_csv).copy()

    required = ["image_path", "center_px", "center_py"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if args.use_selected_point:
        for c in ["px", "py"]:
            if c not in df.columns:
                raise ValueError(f"Missing required selected-point column: {c}")

    rows = []
    for i, row in df.iterrows():
        image_path = str(row["image_path"])

        if args.use_selected_point:
            px = float(row["px"])
            py = float(row["py"])
        else:
            px = float(row["center_px"])
            py = float(row["center_py"])

        east, north = pixel_to_world(image_path, px, py)

        point_id = str(row["bag_index"]) if "bag_index" in row else f"mil_{i:05d}"

        rows.append(
            {
                "point_id": point_id,
                "label": args.target_label,
                "matched_tif": image_path,
                "original_east": east,
                "original_north": north,
            }
        )

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    print("Saved:", args.output_csv)
    print("Rows:", len(out))
    print(out.head())


if __name__ == "__main__":
    main()