import pandas as pd
"""
Merge step1_points.csv and step3_points.csv as following:
step1_points : coordinates of 'center's and 'slide's after grid center point conversion.
step3_points : coordinates that are 'slide's and converted to center after generating center point of the grid.
The script merge step3_points with step1_points based on the ids.
Now all the coordinates of the .csv file is now converted to center point.
"""
# Config
STEP1_CSV = "../coordinate/step1_points.csv"
STEP3_CSV = "../coordinate/step3_points.csv"
OUTPUT_CSV = "../coordinate/final_centered_points.csv"

# Load data
step1 = pd.read_csv(STEP1_CSV)
step3 = pd.read_csv(STEP3_CSV)

# Safety: numeric coords
for df in (step1, step3):
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

# Split step1 into center / slide
step1_center = step1[step1["type"] == "center"].copy()
step1_slide = step1[step1["type"] == "slide"].copy()

# Prepare step3 (slide -> center)
step3_fixed = step3.copy()
step3_fixed["type"] = "center"

# Remove slide rows that were re-centered in step3
slide_ids_fixed = set(step3_fixed["feature_id"])
step1_slide_remaining = step1_slide[
    ~step1_slide["feature_id"].isin(slide_ids_fixed)]

# Combine final result
final_df = pd.concat(
    [
        step1_center,
        step3_fixed,
        step1_slide_remaining
    ],
    ignore_index=True)

# Final cleanup
final_df = final_df.sort_values("feature_id").reset_index(drop=True)

# Save
final_df.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)
print("Total points:", len(final_df))
print("Centers:", (final_df["type"] == "center").sum())
print("Slides (unresolved):", (final_df["type"] == "slide").sum())
