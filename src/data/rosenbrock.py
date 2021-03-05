import GPy
import numpy as np
import pickle
from math import pi
import matplotlib.pyplot as plt
import scipy
import GPy
import numpy as np
import pickle
from math import pi
import matplotlib.pyplot as plt
import scipy


def rosenbrock(num_lf, num_hf, dim):
    def f_high(X):
        return np.array([scipy.optimize.rosen(x) for x in X])[:,None]

    def f_low(X):
        return f_high(X - 0.1) + np.sin(2* np.pi* np.sum(X))

    return get_curve(f_low, f_high, num_lf, num_hf, dim)


def get_curve(f_low, f_high, num_lf, num_hf, dim):

    N = num_lf + num_hf
    N = 500

    X_train = np.random.uniform(np.zeros(dim), np.ones(dim), size=(N, dim))
    X_test = np.random.uniform(np.zeros(dim), np.ones(dim), size=(N, dim))

    X_train_hf = X_train[:num_hf]
    X_train_lf = X_train[num_hf:]

    Y_train_hf = f_high(X_train_hf)
    Y_train_lf = f_low(X_train_lf)

    Y_test = f_high(X_test)

    return X_train_hf, X_train_lf, Y_train_lf, f_high, f_low, X_test, Y_test
