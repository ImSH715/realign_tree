"""
Analyze binary class prototypes and embedding structure.

For BinaryTree:
- 0 = Other
- 1 = Shihuahuaco
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

proto_path = "./outputs/phase2_binary_shihuahuaco/class_prototypes.csv"

df = pd.read_csv(proto_path)

labels = df.iloc[:, 0].astype(str).str.strip()
features = df.iloc[:, 1:].values.astype(np.float32)

label_name = {
    "0": "Other",
    "1": "Shihuahuaco",
    0: "Other",
    1: "Shihuahuaco",
}

display_labels = [label_name.get(x, x) for x in labels.tolist()]

print("=" * 60)
print("Binary prototype analysis")
print("Number of classes:", len(labels))
print("Classes:")
for raw, name in zip(labels, display_labels):
    print(f"  {raw} = {name}")
print("=" * 60)

sim = cosine_similarity(features)

print("\nCosine similarity matrix:")
sim_df = pd.DataFrame(sim, index=display_labels, columns=display_labels)
print(sim_df.round(4))

if len(labels) == 2:
    print("\n" + "=" * 60)
    print("Binary separation")
    print(f"{display_labels[0]} vs {display_labels[1]} similarity: {sim[0, 1]:.4f}")
    print(f"{display_labels[0]} vs {display_labels[1]} distance: {1 - sim[0, 1]:.4f}")

    if sim[0, 1] > 0.95:
        print("Interpretation: prototypes are very close; separation is weak.")
    elif sim[0, 1] > 0.85:
        print("Interpretation: prototypes are moderately close; classifier may work but confusion is expected.")
    else:
        print("Interpretation: prototypes are reasonably separated.")
else:
    print("\nTop confusing classes per prototype:\n")

    for i, label in enumerate(display_labels):
        sims = sim[i].copy()
        sims[i] = -1

        top_idx = np.argsort(sims)[-3:][::-1]

        print(f"\n[{label}]")
        for j in top_idx:
            print(f"  -> {display_labels[j]} (sim={sims[j]:.4f})")