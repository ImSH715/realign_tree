"""
Analyze class prototypes and embedding structure.

Outputs:
- class counts
- prototype distance matrix
- nearest prototype confusion
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load prototypes
# -------------------------
proto_path = "./outputs/phase2_binary_shihuahuaco/class_prototypes.csv"
df = pd.read_csv(proto_path)

labels = df.iloc[:, 0].astype(str).str.strip()
features = df.iloc[:, 1:].values

print("=" * 60)
print("Number of classes:", len(labels))
print("Classes:")
print(labels.tolist())
print("=" * 60)

# -------------------------
# Cosine similarity between prototypes
# -------------------------
sim = cosine_similarity(features)

print("\nTop confusing classes per prototype:\n")

for i, label in enumerate(labels):
    sims = sim[i].copy()
    sims[i] = -1  # ignore self

    top_idx = np.argsort(sims)[-3:][::-1]

    print(f"\n[{label}]")
    for j in top_idx:
        print(f"  -> {labels[j]} (sim={sims[j]:.4f})")

# -------------------------
# Focus: Shihuahuaco
# -------------------------
target = "Shihuahuaco"

if target in labels.values:
    idx = labels[labels == target].index[0]
    sims = sim[idx]

    print("\n" + "=" * 60)
    print(f"[Focus: {target}]")

    sorted_idx = np.argsort(sims)[::-1]

    for j in sorted_idx[:10]:
        print(f"{labels[j]}: {sims[j]:.4f}")
else:
    print(f"{target} not found in prototypes")