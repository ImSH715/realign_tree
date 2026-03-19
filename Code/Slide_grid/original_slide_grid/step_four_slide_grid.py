import pandas as pd
import os

"""
Merge step1_points_lejepa.csv and step3_points_lejepa.csv.
step1_points: Coordinates of 'center' and 'slide' after the initial grid evaluation.
step3_points: Coordinates that were 'slide' and successfully converted to 'center' after the second evaluation.
This script merges step3_points into step1_points based on feature_id.
Now all the coordinates in the resulting .csv file are updated to their final positions.
"""

# ==========================================
# 1. Configuration & Paths
# ==========================================
STEP1_CSV = "step1_points_lejepa.csv"
STEP3_CSV = "step3_points_lejepa.csv"
OUTPUT_CSV = "final_centered_points_lejepa.csv"

def main():
    print("Loading Step 1 and Step 3 CSV files...")
    
    # Load Step 1
    if not os.path.exists(STEP1_CSV):
        raise FileNotFoundError(f"Error: {STEP1_CSV} not found. Please run Step 1.")
    step1 = pd.read_csv(STEP1_CSV)

    # Load Step 3 (Handle case where Step 3 might be empty or missing if no slides were needed)
    if os.path.exists(STEP3_CSV):
        step3 = pd.read_csv(STEP3_CSV)
    else:
        print(f"Warning: {STEP3_CSV} not found. Assuming no points required Step 3 sliding.")
        # Create an empty DataFrame with the expected columns
        step3 = pd.DataFrame(columns=["x", "y", "feature_id", "label", "type"])

    # Ensure coordinates are numeric for safety
    for df in (step1, step3):
        if not df.empty:
            df["x"] = pd.to_numeric(df["x"], errors="coerce")
            df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # ==========================================
    # 2. Data Merging Logic
    # ==========================================
    # Split step1 into center / slide
    step1_center = step1[step1["type"] == "center"].copy()
    step1_slide = step1[step1["type"] == "slide"].copy()

    # Prepare step3 (slide -> center)
    step3_fixed = step3.copy()
    if not step3_fixed.empty:
        step3_fixed["type"] = "center"

    # Remove slide rows from Step 1 that were successfully re-centered in Step 3
    if not step3_fixed.empty:
        slide_ids_fixed = set(step3_fixed["feature_id"])
    else:
        slide_ids_fixed = set()
        
    step1_slide_remaining = step1_slide[~step1_slide["feature_id"].isin(slide_ids_fixed)]

    # Combine final result
    final_df = pd.concat(
        [step1_center, step3_fixed, step1_slide_remaining],
        ignore_index=True
    )

    # Final cleanup (sort by ID for clean output)
    final_df = final_df.sort_values("feature_id").reset_index(drop=True)

    # ==========================================
    # 3. Save Output
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