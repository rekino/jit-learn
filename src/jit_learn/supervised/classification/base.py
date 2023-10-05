import numpy as np
import numpy.typing as ntp


class BaseClassifier(object):

    def fit(self, X: ntp.NDArray, y: ntp.NDArray) -> None:
        self.xdim = X.shape
        self.ydim = y.shape

    def predict(self, X: ntp.NDArray) -> ntp.NDArray:
        assert X.shape == self.xdim, f'X should be of shape {self.xdim}'

        return np.zeros(self.ydim)
