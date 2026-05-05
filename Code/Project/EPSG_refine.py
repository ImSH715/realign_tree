import pandas as pd
import rasterio

input_csv = "outputs/phase3/refined_points.csv"
output_csv = "outputs/phase3/refined_points_world.csv"

df = pd.read_csv(input_csv)

eastings = []
northings = []

for _, row in df.iterrows():
    tif = row["image_path"]
    col = float(row["refined_x"])
    r = float(row["refined_y"])

    with rasterio.open(tif) as src:
        x, y = src.xy(r, col)

    eastings.append(x)
    northings.append(y)

df["refined_east"] = eastings
df["refined_north"] = northings
df.to_csv(output_csv, index=False)

print("Saved:", output_csv)