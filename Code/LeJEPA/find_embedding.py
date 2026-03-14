import numpy as np

data = np.load('data/embedding/lejepa_trained_features.npy', allow_pickle= True)

print(f"Shape: {data.shape}") 
print(f"Dtype: {data.dtype}")

print(f"lable: ", data[:5])