import GPy
import numpy as np
import pickle
from math import pi
import matplotlib.pyplot as plt

np.random.seed(42)


def get_curve1():
    def f_low(t): return np.sin(8 * pi * t)
    def f_high(t): return np.sin(8 * pi * t)**2
    return get_curve(f_low, f_high)


def get_curve2():
    def f_low(t): return np.sin(8 * pi * t)
    def f_high(t): return t**2 * f_low(t)**2
    return get_curve(f_low, f_high)


def get_curve3():
    def f_low(t): return np.sin(8 * pi * t)
    def f_high(t): return t**2 * np.sin(8 * pi * t + pi / 10)**2
    return get_curve(f_low, f_high)


def get_curve4():
    def f_low(t): return np.sin(8 * pi * t)
    def f_high(t): return (t - 1.41) * f_low(t)**2
    return get_curve(f_low, f_high)


def get_curve5():
    def f_low(t): return np.sin(8 * pi * t)
    def f_high(t): return t**2 + np.sin(8 * pi * t + pi/10)
    return get_curve(f_low, f_high)




def get_curve(f_low, f_high):
    f_low = np.vectorize(f_low)
    f_high = np.vectorize(f_high)

    hf_size = 6
    lf_size = 80
    N = lf_size + hf_size

    train_proportion = 0.8

    X = np.linspace(0, 1, N).reshape(-1, 1)
    np.random.shuffle(X)

    X_train = X[:int(N * train_proportion)]
    X_test = X[int(N * train_proportion):]

    X_train_hf = X_train[:hf_size]
    X_train_lf = X_train[hf_size:]

    y_train_hf = f_high(X_train_hf)
    y_train_lf = f_low(X_train_lf)

    y_test = f_high(X_test)

    return X_train_hf, X_train_lf, y_train_lf, f_high, f_low, X_test, y_test
