import numpy as np


def least_confidence(y_pred):
    y_max = np.amax(y_pred, axis=1)
    return np.argsort(y_max), y_max
