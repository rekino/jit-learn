import numpy as np
import numpy.typing as ntp


class BaseClassifier(object):
    def fit(self, X: ntp.NDArray, y: ntp.NDArray) -> None:
        self.X = X
        self.y = y

    def predict(self, X: ntp.NDArray) -> ntp.NDArray:
        if not np.allclose(X, self.X):
            raise NotImplementedError()

        return self.y
