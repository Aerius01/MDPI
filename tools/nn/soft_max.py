import numpy as np


class SoftMax:
    @staticmethod
    def soft_max(y_pred):
        if y_pred == []:
            return []
        else:
            y_pred -= np.max(y_pred)
            softmax = (np.exp(y_pred).T / np.sum(np.exp(y_pred), axis=1)).T
            return softmax
