import numpy as np


class MarginSampling:
    @staticmethod
    def margin_sampling(y_p):
        diff = []
        for i in range(y_p.shape[0]):
            idx = np.argpartition(y_p[i], -2)[-2:]
            subt = np.subtract(y_p[i, idx[0]], y_p[i, idx[1]])
            diff = np.append(diff, np.absolute(subt))
        return np.argsort(diff), diff