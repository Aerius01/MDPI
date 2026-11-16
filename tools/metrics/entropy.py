import numpy as np


def entropy(y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices sorted by descending entropy, per-sample entropies in sample order)."""
    y_log = np.log(y_pred)
    y_log[y_log == -float('inf')] = -10000000
    ent = -np.sum(np.multiply(y_pred, y_log), 1)
    return np.argsort(-ent), ent
