import numpy as np


def entropy(y_pred):
    y_log = np.log(y_pred)
    y_log[y_log == -float('inf')] = -10000000
    ent = -np.sum(np.multiply(y_pred, y_log), 1)
    return np.argsort(-ent), ent[np.argsort(-ent)]