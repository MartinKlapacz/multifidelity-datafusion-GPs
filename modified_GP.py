import GPy
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from abc import ABC, abstractmethod
import pickle
from NARGP_kernel import NARPGKernel
import random
from math import pi


def generate_random_training_data(a, b):
    X = np.linspace(a, b, 10).reshape(-1, 1)
    Y = np.array([np.round(np.sin(i), 2) for i in X]).reshape(-1, 1)
    return X, Y


def generate_Mauna_Loa_data(reduce_by: int = 1):
    with open("./datasets/mauna_loa", 'rb') as fid:
        data = pickle.load(fid)
    X_train = data.get('X')[::reduce_by]
    y_train = data.get('Y')[::reduce_by]
    X_test = data.get('Xtest')[::reduce_by]
    y_test = data.get('Ytest')[::reduce_by]
    return X_train, y_train, X_test, y_test


class GPDataAugmentation:

    def __init__(self, tau: float, n: int, input_dims: int):
        '''
        input: lff
            low fidelity function
        input: tau
            distance to neighbour points used in taylor expansion
        input n: 
            number of lags in the augmentation process (2*n + 1)
        '''
        self.tau = tau
        self.n = n
        self.input_dims = input_dims

        self.lf_model = None
        self.lf_X = None
        self.lf_Y = None
        self.__lf_mean_predict = None

        self.hf_model = None
        self.hf_X = None
        self.hf_Y = None
        self.hf_mean_predict = None

    def lf_fit(self, lf_X, lf_Y):
        assert lf_X.ndim == 2
        self.lf_X = lf_X
        self.lf_Y = lf_Y
        self.lf_model = GPy.models.GPRegression(
            X=lf_X, Y=lf_Y, initialize=True
        )
        self.lf_model.optimize()
        self.__lf_mean_predict = lambda t: self.lf_model.predict(np.array([t]))[0][0]

    def hf_fit(self, hf_X, hf_Y):
        assert self.lf_model is not None, "low fidelity model musst be initialized"
        assert hf_X.ndim == 2
        augmented_hf_X = self.__augment_vector_list(hf_X)
        self.hf_model = GPy.models.GPRegression(
            X=augmented_hf_X, Y=hf_Y, initialize=True
        )
        self.hf_model.optimize()  # ARD
        self.hf_mean_predict = lambda t: self.hf_model.predict(np.array([t]))[0]

    def predict(self, x):
        assert x.ndim == 1, 'prediction input must be vector'
        assert len(x) == self.input_dims, 'invalid input dimension'
        return self.hf_mean_predict(x)

    def plot(self, a, b):
        assert a < b, "b must be greater than a"
        assert self.hf_model is not None, 'model is not fitted yet'

        X = np.linspace(a, b, (b - a) * 10)
        Y = [self.predict(x) for x in X]
        plt.plot(X, Y, 'ro')
        plot.show()

    def __augment_vector(self, x: np.ndarray):
        assert x.ndim == 1, 'vector to augment must have vector shape'

        # augment x with its low-fidelity prediction value
        _x = x.copy()
        foo = self.__lf_mean_predict(_x)
        x = np.concatenate([_x, foo])
        # if n > 0 include lagged values
        for i in range(1, self.n+1):
            x = np.concatenate([x, self.__lf_mean_predict(_x + i * self.tau)])
            x = np.concatenate([x, self.__lf_mean_predict(_x - i * self.tau)])
        return x

    def __augment_vector_list(self, X):
        assert isinstance(X, np.ndarray), 'input must be an array'
        assert len(X) > 0, 'input must be non-empty'

        augmented_X = np.array([self.__augment_vector(x) for x in X])
        assert augmented_X.shape[1] == self.input_dims + 2 * self.n + 1
        return augmented_X


if __name__ == "__main__":
    def f_high(t): return t**2 - np.sin(8 * pi * t - pi / 10)**2
    def f_low(t): return np.sin(8 * pi * t)

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

    model = GPDataAugmentation(tau=.5, n=1, input_dims=1)
    model.lf_fit(lf_X=X_train_lf, lf_Y=y_train_lf)
    model.hf_fit(hf_X=X_train_hf, hf_Y=y_train_hf)
    model.predict()


    # model.hf_fit(X=X_train_hf, lf_Y=y_train_hf)
    # predictions = model.predict(X=X_test)

# TODO abstract class, multiple implementations of GP Methods inheriting of abstract class
# TODO child classes:
#   GP_augmented_data, select better kernel than RBF
#   NARGP
#   MFDGP
# TODO isCheap parameter
# TODO store plots
# TODO make tau independent of input dim
