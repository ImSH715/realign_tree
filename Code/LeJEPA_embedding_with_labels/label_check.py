import numpy as np
import geopandas as gpd
from collections import Counter

# --- 경로 설정 ---
SHP_PATH = r"Z:\ai4eo\Shared\2025_Turing_L\Project\Annotated tree centroids\trees_32718.shp"
LABEL_NPY_PATH = "data/label/labels.npy"

def check_labels():
    print("=== 데이터 라벨 검증 ===")
    
    # 1. 원본 Shapefile 데이터 개수 확인
    try:
        gdf = gpd.read_file(SHP_PATH)
        print(f"1. 원본 Shapefile 내 전체 나무(포인트) 개수: {len(gdf)}개")
    except Exception as e:
        print(f"Shapefile을 읽는 중 오류 발생: {e}")
        return

    # 2. 추출되어 저장된 npy 라벨 개수 확인
    try:
        labels = np.load(LABEL_NPY_PATH)
        print(f"2. 실제 추출되어 저장된 라벨(패치) 개수: {len(labels)}개")
    except Exception as e:
        print(f"labels.npy 파일을 읽는 중 오류 발생 (아직 데이터 추출이 안 끝났을 수 있습니다): {e}")
        return

    print("-" * 30)
    
    # 3. 누락된 개수 계산
    missing = len(gdf) - len(labels)
    if missing > 0:
        print(f"※ 원본 대비 추출되지 않은 데이터 수: {missing}개")
        print("   (이유: 일부 포인트가 .tif 이미지 범위를 벗어났거나, 이미지 가장자리에 너무 붙어있어 448x448 패치로 자를 수 없어서 제외되었을 수 있습니다.)")
    elif missing < 0:
        print(f"※ 원본 대비 추출된 데이터가 더 많습니다: {-missing}개")
        print("   (이유: 이미지들이 서로 겹치는(Overlap) 구역이 있어서, 같은 나무가 여러 번 추출되었을 수 있습니다.)")
    else:
        print("※ 완벽하게 모든 데이터가 1:1로 추출되었습니다!")

    print("-" * 30)
    
    # 4. 라벨 종류별 분포 확인 (어떤 수종/ID가 몇 개씩 있는지)
    print("3. 추출된 라벨 세부 분포:")
    label_counts = Counter(labels)
    
    # 가장 많이 추출된 상위 10개만 출력 (너무 길어지는 것 방지)
    for label, count in label_counts.most_common(10):
        print(f"   - {label}: {count}개")
        
    if len(label_counts) > 10:
        print(f"   ... (외 {len(label_counts) - 10}종의 라벨이 더 있습니다)")

if __name__ == "__main__":
    check_labels()