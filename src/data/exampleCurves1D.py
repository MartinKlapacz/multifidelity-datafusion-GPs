import GPy
import numpy as np
import pickle
from math import pi
import matplotlib.pyplot as plt

np.random.seed(42)


def get_curve1(num_hf, num_lf):
    def f_low(t): return np.sin(8 * pi * t)
    def f_high(t): return np.sin(8 * pi * t)**2
    return get_curve(f_low, f_high, num_hf, num_lf)


def get_curve2(num_hf, num_lf):
    def f_low(t): return np.sin(8 * pi * t)
    def f_high(t): return t**2 * f_low(t)**2
    return get_curve(f_low, f_high, num_hf, num_lf)


def get_curve3(num_hf, num_lf):
    """phase shifted oscillations"""
    def f_low(t): return np.sin(8 * pi * t)
    def f_high(t): return t**2 + np.sin(8 * pi * t + pi / 10)**2
    return get_curve(f_low, f_high, num_hf, num_lf)


def get_curve4(num_hf, num_lf):
    def f_low(t): return np.sin(8 * pi * t)
    def f_high(t): return (t - 1.41) * f_low(t)**2
    return get_curve(f_low, f_high, num_hf, num_lf)


def get_curve5(num_hf, num_lf):
    """different periodicities"""
    def f_low(t): return np.sin(6 * np.sqrt(2) * np.pi * t)
    def f_high(t): return np.sin(8 * pi * t + pi/10)
    return get_curve(f_low, f_high, num_hf, num_lf)


def get_discontinuity1(num_hf, num_lf):
    def f_low(t):
        return .5*(6*t - 2)**2 * np.sin(12*t - 4) + 10*(t - .5)
    def f_high(t):
        return 2 * f_low(t) - 20 * t + 20
    return get_curve(f_low, f_high, num_hf, num_lf)

def get_discontinuity2(num_hf, num_lf):
    # linear relation
    def f_high(t): 
        if t < .3:
            return np.sin(30*t)
        elif t < .35:
            return t * 20 - 5
        else:
            return np.sin(49 * t) + 6
    def f_low(t):
        return 2 * f_high(t) + 3
    return get_curve(f_low, f_high, num_hf, num_lf)

def get_discontinuity3(num_hf, num_lf):
    # simple nonlinear relation
    def f_high(t): 
        if t < .3:
            return np.sin(30*t)
        elif t < .35:
            return t * 20 - 5
        else:
            return np.sin(49 * t) + 6
    def f_low(t):
        return 2 * f_high(t) + t
    return get_curve(f_low, f_high, num_hf, num_lf)


def get_discontinuity4(num_hf, num_lf):
    # quadratic nonlinearity
    def f_high(t): 
        if t < .3:
            return np.sin(30*t)
        elif t < .35:
            return t * 20 - 5
        else:
            return np.sin(49 * t) + 6
    def f_low(t):
        return 2 * f_high(t) + t**2
    return get_curve(f_low, f_high, num_hf, num_lf)

def get_discontinuity5(num_hf, num_lf):
    # highly nonlinear -> bad performance
    def f_high(t): 
        if t < .3:
            return np.sin(30*t)
        elif t < .35:
            return t * 20 - 5
        else:
            return np.sin(49 * t) + 6
    def f_low(t):
        return 2 * f_high(t) * t**2 + np.sin(1 / (t+1))
    return get_curve(f_low, f_high, num_hf, num_lf)
    

def get_curve(f_low, f_high, num_hf, num_lf):
    f_low = np.vectorize(f_low)
    f_high = np.vectorize(f_high)

    N = num_lf + num_hf

    train_proportion = 0.8

    X = np.linspace(0, 1, N)[:,None]
    np.random.shuffle(X)

    X_train = X[:int(N * train_proportion)]
    X_test = X[int(N * train_proportion):]

    X_train_hf = X_train[:num_hf]
    X_train_lf = X_train[num_hf:]

    y_train_hf = f_high(X_train_hf)
    y_train_lf = f_low(X_train_lf)

    y_test = f_high(X_test)
    assert len(X_train_hf) < len(X_train_lf)
    return X_train_hf, X_train_lf, y_train_lf, f_high, f_low, X_test, y_test
