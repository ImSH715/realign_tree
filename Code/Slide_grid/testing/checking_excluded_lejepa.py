import os
import glob
import rasterio
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm

# --------------------------
# Configuration & Paths
# --------------------------
BASE_DIR = r"/mnt/parscratch/users/acb20si/2025_Forge/OSINFOR_data/01. Ortomosaicos/2023"
ANNOTATED_COR = r"/mnt/parscratch/users/acb20si/label_tree_shp/trees_32718.shp"

OUTPUT_DIR = "data/label"
OUTPUT_MISSING_SHP = os.path.join(OUTPUT_DIR, "missing_points.shp")
OUTPUT_FOOTPRINTS_SHP = os.path.join(OUTPUT_DIR, "valid_tif_footprints.shp") # 정상 TIF 영역

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading original shapefile: {ANNOTATED_COR}")
    gdf = gpd.read_file(ANNOTATED_COR)
    
    if 'temp_id' not in gdf.columns:
        gdf['temp_id'] = range(len(gdf))
        
    total_points = len(gdf)
    tif_files = glob.glob(os.path.join(BASE_DIR, "2023-*", "*.tif"))
    
    if not tif_files:
        print("Error: No TIF files found.")
        return

    extracted_ids = set()
    error_files = []
    valid_footprints = [] # 정상 TIF들의 영역(Bounding Box)을 모아둘 리스트

    print("\nScanning TIF bounds to find intersecting points...")
    for tif_path in tqdm(tif_files, desc="Checking TIFs"):
        try:
            with rasterio.open(tif_path) as src:
                bounds = src.bounds
                img_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
                bbox_gdf = gpd.GeoDataFrame({'geometry': [img_box], 'filename': os.path.basename(tif_path)}, crs=src.crs)
                
                # 좌표계 통일
                if bbox_gdf.crs != gdf.crs:
                    bbox_gdf = bbox_gdf.to_crs(gdf.crs)

                # 1. 정상적으로 읽힌 TIF의 영역을 리스트에 저장
                valid_footprints.append(bbox_gdf)

                # 2. 해당 영역 안에 들어오는 포인트 ID 수집
                intersecting = gdf[gdf.geometry.intersects(bbox_gdf.geometry.iloc[0])]
                for _, row in intersecting.iterrows():
                    extracted_ids.add(row['temp_id'])
                    
        except Exception as e:
            error_files.append((os.path.basename(tif_path), str(e)))
            continue

    # --- Out of Bounds 결과 계산 ---
    missing_gdf = gdf[~gdf['temp_id'].isin(extracted_ids)]
    
    # 발자국(Footprints) 병합
    if valid_footprints:
        footprints_gdf = pd.concat(valid_footprints, ignore_index=True)
        # 전체 정상 TIF들을 하나로 합친 거대한 다각형 생성 (Out of Bounds 판별용)
        total_valid_area = footprints_gdf.unary_union 
        
        # 진짜 Out of bounds인지 검사: 합쳐진 영역 바깥에 있는 포인트들만 필터링
        out_of_bounds_mask = ~missing_gdf.geometry.intersects(total_valid_area)
        out_of_bounds_count = out_of_bounds_mask.sum()
    else:
        out_of_bounds_count = len(missing_gdf)

    print("\n" + "="*50)
    print("✨ Scanning Complete ✨")
    print("="*50)
    print(f"Total Points: {total_points}")
    print(f"Found (Inside valid TIFs): {len(extracted_ids)}")
    print(f"Missing Points: {len(missing_gdf)}")
    print(f"  -> 이 중 완전히 정상 TIF 영역 밖(Out of Bounds)인 개수: {out_of_bounds_count}")
    
    if error_files:
        print(f"\nWarning: {len(error_files)} TIF 파일 에러 발생 (영역 확인 불가):")
        for err_file, err_msg in error_files:
            print(f"  - {err_file}")
            
    # 결과 파일 저장
    if not missing_gdf.empty:
        missing_gdf.to_file(OUTPUT_MISSING_SHP)
    if valid_footprints:
        footprints_gdf.to_file(OUTPUT_FOOTPRINTS_SHP)
        
    print(f"\n파일 저장 완료!")
    print(f"  - 누락된 포인트: {OUTPUT_MISSING_SHP}")
    print(f"  - 정상 TIF 바운딩 박스: {OUTPUT_FOOTPRINTS_SHP}")
    print("="*50)

if __name__ == "__main__":
    main()