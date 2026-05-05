import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

df = pd.read_csv("./data/valid_points_binary.csv")

gdf = gpd.GeoDataFrame(
    df,
    geometry=[Point(xy) for xy in zip(df["x_epsg32718"], df["y_epsg32718"])],
    crs="EPSG:32718"
)

gdf.to_file("./data/valid_points_binary.shp")

print("Saved SHP")