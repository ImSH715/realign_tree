import numpy as np
import os

def main():
    embedding_dir = "data/embeddings"
    label_dir = "data/label"

    print("Loading all split datasets...")
    
    # 1. Load Embeddings
    train_emb = np.load(os.path.join(embedding_dir, "train_embeddings.npy"))
    val_emb = np.load(os.path.join(embedding_dir, "val_embeddings.npy"))
    test_emb = np.load(os.path.join(embedding_dir, "test_embeddings.npy"))
    
    # 2. Load Labels
    train_lbl = np.load(os.path.join(label_dir, "train_labels.npy"))
    val_lbl = np.load(os.path.join(label_dir, "val_labels.npy"))
    test_lbl = np.load(os.path.join(label_dir, "test_labels.npy"))

    print(f"Shapes before merge -> Train: {train_emb.shape}, Val: {val_emb.shape}, Test: {test_emb.shape}")

    # 3. Concatenate
    combined_emb = np.concatenate([train_emb, val_emb, test_emb], axis=0)
    combined_lbl = np.concatenate([train_lbl, val_lbl, test_lbl], axis=0)

    # label counting
    assert len(combined_emb) == len(combined_lbl), "Mismatch in combined lengths!"

    # 4. Save
    out_emb_path = os.path.join(embedding_dir, "combined_embeddings.npy")
    out_lbl_path = os.path.join(label_dir, "combined_labels.npy")
    
    np.save(out_emb_path, combined_emb)
    np.save(out_lbl_path, combined_lbl)

    print("\nSuccessfully Combined!")
    print(f"Total Combined Samples: {len(combined_emb)}")
    print(f"Saved to:\n - {out_emb_path}\n - {out_lbl_path}")

if __name__ == "__main__":
    main()