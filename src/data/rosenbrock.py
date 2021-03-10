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
        return f_high(X - 0.1) + np.sin(2* np.pi* np.sum(X**2))*0.1

    return get_curve(f_low, f_high, num_lf, num_hf, dim)

def product_sin2(num_lf, num_hf):
    a = [2.2 * np.pi, np.pi]

    def hf_2d(param):
        x = np.atleast_2d(param)
        return np.sin(x[:, 0] * a[0]) * np.sin(x[:, 1] * a[1])


    def lf_2d(param):
        x = np.atleast_2d(param)
        return hf_2d(x) - 1.2 * (np.sin(x[:,0] * np.pi *0.1) + np.sin(x[:,1] * np.pi *0.1))

    f_high = lambda x: np.atleast_2d(hf_2d(x)).T
    f_low = lambda x: np.atleast_2d(lf_2d(x)).T

    return get_curve(f_low, f_high, num_lf, num_hf, 2)

def product_sin4(num_lf, num_hf):
    a = [np.pi, np.pi, np.pi, np.pi]

    def hf_4d(param):
        x = np.atleast_2d(param)
        return np.sin(x[:, 0] * a[0]) * np.sin(x[:, 1] * a[1]) * np.sin(x[:, 2] * a[2]) * np.sin(x[:, 3] * a[3])


    def lf_4d(param):
        x = np.atleast_2d(param)
        return hf_4d(x) - 0.25 * (np.sin(x[:,0] * np.pi * 0.1) + np.sin(x[:,1] * np.pi *0.05)
                              + np.sin(x[:, 2] * 0.15 * np.pi) + np.sin(x[:, 3] * 0.2 * np.pi))

    f_high = lambda x: np.atleast_2d(hf_4d(x)).T
    f_low = lambda x: np.atleast_2d(lf_4d(x)).T

    return get_curve(f_low, f_high, num_lf, num_hf, 4)

def product_sin8(num_lf, num_hf):
    a = [np.pi] * 8

    def hf_8d(param):
        X = np.atleast_2d(param)
        temp = []
        for x in X: 
            temp.append(np.prod([np.sin(x[i]) * a[i] for i in range(8)], axis=0))
        return np.array(temp)


    def lf_8d(param):
        X = np.atleast_2d(param)
        temp = []
        for x in X: 
            temp.append(np.prod([np.sin(x[i]) * a[i] * 0.2 for i in range(8)], axis=0))
        return hf_8d(x) - np.array(temp)*0.25

    f_high = lambda x: np.atleast_2d(hf_8d(x)).T
    f_low = lambda x: np.atleast_2d(lf_8d(x)).T

    return get_curve(f_low, f_high, num_lf, num_hf, 8)

def get_curve(f_low, f_high, num_lf, num_hf, dim):

    N = num_lf + num_hf

    X_train = np.random.uniform(np.zeros(dim), np.ones(dim), size=(N, dim))
    X_test = np.random.uniform(np.zeros(dim), np.ones(dim), size=(500, dim))

    X_train_hf = X_train[:num_hf]
    X_train_lf = X_train[num_hf:]

    Y_train_hf = f_high(X_train_hf)
    Y_train_lf = f_low(X_train_lf)

    Y_test = f_high(X_test)

    return X_train_hf, X_train_lf, Y_train_lf, f_high, f_low, X_test, Y_test
