import numpy as np


def euclidean_distance(row1: np.ndarray, row2: np.ndarray) -> float:
    return np.linalg.norm(row1 - row2)


def manhattan_distance(row1: np.ndarray, row2: np.ndarray) -> float:
    return (np.absolute(row1 - row2)).sum()


def custom_accuracy(row1: np.ndarray, row2: np.ndarray) -> float:
    error = []
    for i in range(row1.shape[0]):
        error.append(100.0 - (np.absolute(row2[i] - row1[i]) * 25.0))
    return float(np.mean(error))


def chebyshev_distance(row1: np.ndarray, row2: np.ndarray) -> float:
    return (np.absolute(row1 - row2)).max()


def cosine_distance(row1: np.ndarray, row2: np.ndarray) -> float:
    return np.dot(row1, row2) / (np.sqrt((row1 ** 2).sum()) + np.sqrt((row2 ** 2).sum()))
