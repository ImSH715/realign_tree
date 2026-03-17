import rasterio
import geopandas as gpd
from shapely.geometry import box
import os
from glob import glob

# Directory Configuration
base_dir = r"Z:\ai4eo\Shared\2025_Forge\OSINFOR_data\01. Ortomosaicos\2023"
output_dir = r"shapefileTIF"
target_crs = "EPSG:32718"

# Find all .tif files
tif_files = glob(os.path.join(base_dir, '2023-*', '*.tif'))

# Create output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Directory created: {output_dir}")

def generate_bbox_shapefiles(tif_paths, save_path):
    success_count = 0
    error_files = []

    print("=" * 85)
    print(f"{'Source TIF File':<55} | {'Status'}")
    print("=" * 85)

    for tif_path in tif_paths:
        file_name = os.path.basename(tif_path)
        try:
            with rasterio.open(tif_path) as src:
                # 1. 이미지의 네 모서리 좌표(좌측 하단, 우측 상단)만 추출
                # src.bounds는 (left, bottom, right, top) 형태의 튜플을 반환합니다.
                bounds = src.bounds
                
                # 2. Bounding Box로 단순한 사각형 폴리곤 생성 (디테일 무시)
                bbox_geom = box(*bounds)
                
                # 3. 원래 좌표계(CRS)를 적용하여 GeoDataFrame 생성
                gdf_footprint = gpd.GeoDataFrame(
                    {'file_name': [file_name]}, 
                    geometry=[bbox_geom], 
                    crs=src.crs
                )
                
                # 4. 목표 CRS(EPSG:32718)로 변환
                if gdf_footprint.crs != target_crs:
                    gdf_footprint = gdf_footprint.to_crs(target_crs)
                
                # 5. Shapefile로 저장
                file_base = os.path.splitext(file_name)[0]
                output_file = os.path.join(save_path, f"{file_base}_bbox.shp")
                
                gdf_footprint.to_file(output_file)
                print(f"{file_name[:55]:<55} | Success (Bounding Box)")
                success_count += 1

        except Exception as e:
            error_files.append((file_name, str(e)))
            print(f"{file_name[:55]:<55} | FAILED")
            continue

    print("-" * 85)
    print(f"Total Successfully Processed : {success_count}")
    print(f"Total Failed                 : {len(error_files)}")
    print("=" * 85)

# Run the process
generate_bbox_shapefiles(tif_files, output_dir)