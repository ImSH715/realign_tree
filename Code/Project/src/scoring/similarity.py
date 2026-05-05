import numpy as np


def cosine_similarity_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    y_norm = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
    return x_norm @ y_norm.T


def euclidean_similarity_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x ** 2, axis=1, keepdims=True)
    y2 = np.sum(y ** 2, axis=1, keepdims=True).T
    dist2 = x2 + y2 - 2 * (x @ y.T)
    dist2 = np.maximum(dist2, 0.0)
    dist = np.sqrt(dist2)
    return -dist