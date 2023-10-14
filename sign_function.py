import numpy as np
import matplotlib.pyplot as plt

from src.jit_learn import BaseClassifier

x_train = np.linspace(-1, 1)
y_train = np.sign(x_train)

cls = BaseClassifier(1, 46)
cls.fit(x_train[:, None], y_train)

x_test = np.linspace(-1, 1, 1000)

y_bounds = np.asarray([cls.predict(_x) for _x in x_test])

plt.plot(x_test, y_bounds[:, 0])
plt.plot(x_test, y_bounds[:, 1])

plt.show()
