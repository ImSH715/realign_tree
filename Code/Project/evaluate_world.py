import pandas as pd
import numpy as np

df = pd.read_csv("outputs/phase3/refined_points_world.csv")

# expected columns:
# gt_east, gt_north = ground truth answer
# original_east, original_north = initial input point
# refined_east, refined_north = output point

before = np.sqrt((df["original_east"] - df["gt_east"])**2 + (df["original_north"] - df["gt_north"])**2)
after = np.sqrt((df["refined_east"] - df["gt_east"])**2 + (df["refined_north"] - df["gt_north"])**2)

improved = (after < before).sum()
unchanged = (after == before).sum()
worse = (after > before).sum()

print("N:", len(df))
print("Mean distance before:", before.mean())
print("Mean distance after :", after.mean())
print("Median distance before:", np.median(before))
print("Median distance after :", np.median(after))
print("Improved:", improved)
print("Unchanged:", unchanged)
print("Worse:", worse)

for th in [2, 5, 10]:
    acc_before = (before <= th).mean()
    acc_after = (after <= th).mean()
    print(f"Accuracy within {th} m - before: {acc_before:.3f}, after: {acc_after:.3f}")