import GPy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from abc import ABC, abstractmethod
import pickle
from NARGP_kernel import NARPGKernel
from datasets import get_example_data
import random
from math import pi


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
        self.lf_X, self.lf_Y = None, None
        self.__lf_mean_predict = None

        self.hf_model = None

    def lf_fit(self, lf_X, lf_Y):
        assert lf_X.ndim == 2
        self.lf_X, self.lf_Y = lf_X, lf_Y
        self.lf_model = GPy.models.GPRegression(
            X=lf_X, Y=lf_Y, initialize=True
        )
        self.lf_model.optimize()
        self.__lf_mean_predict = lambda t: self.lf_model.predict(np.array([t]))[
            0][0]

    def hf_fit(self, hf_X, hf_Y):
        assert self.lf_model is not None, "low fidelity model must be initialized"
        assert hf_X.ndim == 2
        self.hf_X, self.hf_Y = hf_X, hf_Y
        augmented_hf_X = self.__augment_vector_list(hf_X)
        self.hf_model = GPy.models.GPRegression(
            X=augmented_hf_X, Y=hf_Y, initialize=True
        )
        self.hf_model.optimize()  # ARD

    def predict(self, X_test):
        assert X_test.ndim == 2
        assert X_test.shape[1] == self.input_dims
        X_test = self.__augment_vector_list(X_test)
        return self.hf_model.predict(X_test)

    def predict_means(self, X_test):
        return self.predict(X_test)[0]

    def plot(self):
        assert self.input_dims == 1, '2d plots need one-dimensional data'
        assert self.hf_model is not None, 'model is not fitted yet'

        a, b = np.min(self.lf_X), np.max(self.lf_X)

        X = np.linspace(a, b, 1000)
        predictions = self.predict_means(X.reshape(-1, 1))

        plt.plot(self.lf_X, self.lf_Y, 'ro', label='low-fidelity')
        plt.plot(self.hf_X, self.hf_Y, 'bo', label='high-fidelity')
        plt.plot(X, predictions, label='prediction', linestyle='dashed')
        plt.legend()
        plt.show()

    def __augment_vector(self, x: np.ndarray):
        assert x.ndim == 1, 'vector to augment must have vector shape'

        # augment x with its low-fidelity prediction value
        _x = x.copy()
        x = np.concatenate([_x, self.__lf_mean_predict(_x)])
        # if n > 0 include lagged values
        for i in range(1, self.n+1):
            x = np.concatenate([x, self.__lf_mean_predict(_x + i * self.tau)])
            x = np.concatenate([x, self.__lf_mean_predict(_x - i * self.tau)])
        return x

    def __augment_vector_list(self, X):
        assert isinstance(X, np.ndarray), 'input must be an array'
        assert len(X) > 0, 'input must be non-empty'

        augmented_X = np.array([self.__augment_vector(x) for x in X])
        self.hf_input_dim = augmented_X.shape[1]
        assert augmented_X.shape[1] == self.input_dims + 2 * self.n + 1
        return augmented_X


if __name__ == "__main__":
    X_train_hf, X_train_lf, y_train_hf, y_train_lf, X_test, y_test = get_example_data()

    # create, train, test model
    model = GPDataAugmentation(tau=.01, n=1, input_dims=1)
    model.lf_fit(lf_X=X_train_lf, lf_Y=y_train_lf)
    model.hf_fit(hf_X=X_train_hf, hf_Y=y_train_hf)
    predictions = model.predict_means(X_test)
    mse = mean_squared_error(y_true=y_test, y_pred=predictions)
    print('mean squared error: {}'.format(mse))
    model.plot()


# TODO abstract class, multiple implementations of GP Methods inheriting of abstract class
# TODO child classes:
# TODO plot method
#   GP_augmented_data, select better kernel than RBF
#   NARGP
#   MFDGP
# TODO isCheap parameter
