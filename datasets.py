import GPy
import numpy as np
import pickle
import random
from math import pi



def get_Mauna_Loa_data(reduce_by: int = 1):
    with open("./datasets/mauna_loa", 'rb') as fid:
        data = pickle.load(fid)
    X_train = data.get('X')[::reduce_by]
    y_train = data.get('Y')[::reduce_by]
    X_test = data.get('Xtest')[::reduce_by]
    y_test = data.get('Ytest')[::reduce_by]
    return X_train, y_train, X_test, y_test

def get_example_data():
    # define fidelity models
    def f_high(t): return t**2 - np.sin(8 * pi * t - pi / 10)**2
    def f_low(t): return np.sin(8 * pi * t)

    # prepare data
    hf_size = 20
    lf_size = 80
    train_size = 80
    test_size = 20
    N = lf_size + hf_size

    X = np.linspace(0, 1, N).reshape(-1, 1)
    random.shuffle(X)

    X_train = X[:train_size]
    X_test = X[train_size:]

    X_train_hf = X_train[:hf_size]
    X_train_lf = X_train[hf_size:]

    y_train_hf = np.array([f_high(t) for t in X_train_hf])
    y_train_lf = np.array([f_high(t) for t in X_train_lf])

    y_test = np.array([f_high(t) for t in X_test])

    return X_train_hf, X_train_lf, y_train_hf, y_train_lf, X_test, y_test