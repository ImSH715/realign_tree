import numpy as np
import os

embedding_path = "data/embeddings/embeddings.npy"

if os.path.exists(embedding_path):
    # Load data
    embeddings = np.load(embedding_path)

    print(f"Embedding Information")
    print(f"File location: {embedding_path}")
    
    # Use the 'embeddings' variable here, not 'embedding_path'
    print(f"Total number of rows : {embeddings.shape[0]}")
    print(f"Embedding dimension (columns) : {embeddings.shape[1]}")
    print(f"Data Type: {embeddings.dtype}")
    print(f"Total Shape: {embeddings.shape}")
else:
    print(f"Cannot find embeddings at: {embedding_path}")