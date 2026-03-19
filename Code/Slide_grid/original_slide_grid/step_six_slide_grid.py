import pandas as pd
import os

# ==========================================
# 1. Configuration & Paths
# ==========================================
STEP1_CSV = "step1_points_lejepa.csv"  # 1차 시도 결과
STEP3_CSV = "step3_points_lejepa.csv"  # 2차 시도 결과
STEP5_CSV = "step5_points_lejepa.csv"  # 3차 시도 결과 (방금 한 것)
OUTPUT_CSV = "final_centered_points_lejepa.csv"

def main():
    print("Loading CSV files for final merge...")
    
    # 순서대로 리스트에 담습니다. (뒤로 갈수록 최신 상태의 좌표입니다)
    dfs = []
    
    for file_path in [STEP1_CSV, STEP3_CSV, STEP5_CSV]:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if not df.empty:
                df["x"] = pd.to_numeric(df["x"], errors="coerce")
                df["y"] = pd.to_numeric(df["y"], errors="coerce")
                dfs.append(df)

    if not dfs:
        print("No data found to merge.")
        return

    # 1. 모든 데이터를 위아래로 이어 붙입니다.
    combined = pd.concat(dfs, ignore_index=True)

    # 2. 'feature_id'가 중복될 경우, 가장 마지막에 등장하는 행(가장 최신 Step의 좌표)만 남깁니다.
    final_df = combined.drop_duplicates(subset=["feature_id"], keep="last")
    
    # 3. ID 순서대로 정렬하여 예쁘게 정리합니다.
    final_df = final_df.sort_values("feature_id").reset_index(drop=True)

    # ==========================================
    # 4. Save Output
    # ==========================================
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved final merged dataset to: {OUTPUT_CSV}")
    print("-" * 40)
    print(f"Total points processed: {len(final_df)}")
    print(f"Successfully Centered: {(final_df['type'] == 'center').sum()}")
    print(f"Unresolved Slides (Failed to center): {(final_df['type'] == 'slide').sum()}")
    print("-" * 40)

if __name__ == "__main__":
    main()