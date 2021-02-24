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
        return 1.5 * f_high(X) + 3 #
        + X[:,1] * .01
    return get_curve(f_low, f_high, num_lf, num_hf, dim)


def get_curve(f_low, f_high, num_lf, num_hf, dim):

    hf_size = num_hf
    lf_size = num_lf
    N = lf_size + hf_size

    train_proportion = 0.8

    X = np.random.uniform(-5*np.ones(dim),5*np.ones(dim), size=(N, dim))

    np.random.shuffle(X)

    X_train = X[:int(N * train_proportion)]
    X_test = X[int(N * train_proportion):]

    X_train_hf = X_train[:hf_size]
    X_train_lf = X_train[hf_size:]

    y_train_hf = f_high(X_train_hf)
    y_train_lf = f_low(X_train_lf)

    y_test = f_high(X_test)

    return X_train_hf, X_train_lf, y_train_lf, f_high, f_low, X_test, y_test
