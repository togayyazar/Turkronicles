from typing import Iterable

import numpy as np


def cosine_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    if X.ndim == 1 and Y.ndim == 1:
        return np.dot(X / np.linalg.norm(X), Y / np.linalg.norm(Y))

    if X.ndim == 1:
        X = X[np.newaxis, :]
    if Y.ndim == 1:
        Y = Y[np.newaxis, :]

    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return np.dot(X, Y.T)


def jaccard_similarity(source: Iterable, target: Iterable) -> float:
    return len(set(source).intersection(target)) / len(set(source).union(set(target)))
