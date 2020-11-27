import GPy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import abc
import pickle
from NARGP_kernel import NARPGKernel
from datasets import get_example_data
import random
from math import pi

class AbstractGP(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def predict(self, X_test):
        pass

    @abc.abstractmethod
    def plot(self):
        pass

class GPDataAugmentation(AbstractGP):

    def __init__(self, tau: float, n: int, input_dims: int, f_low: callable=None):
        '''
        input: tau
            distance to neighbour points used in taylor expansion
        input n: 
            number of derivatives which will be included when training the high-fidelity model,
            adds 2*n+1 dimensions to the high-fidelity training data
        input input_dims:
            dimensionality of the input data
        input f_low:
            closed form of a low-fidelity prediction function, 
            if not provided, call self.lf_fit() to train a low-fidelity GP which will be used for low-fidelity predictions instead
        '''
        self.tau = tau
        self.n = n
        self.input_dims = input_dims

        self.lf_model = None
        self.lf_X, self.lf_Y = None, None
        self.__lf_mean_predict = f_low

        self.hf_model = None

    def lf_fit(self, lf_X, lf_Y):
        print(self.__lf_mean_predict)
        assert self.__lf_mean_predict is None, 'low-fidelity already specified'
        assert lf_X.ndim == 2
        self.lf_X, self.lf_Y = lf_X, lf_Y
        self.lf_model = GPy.models.GPRegression(
            X=lf_X, Y=lf_Y, initialize=True
        )
        self.lf_model.optimize()
        self.__lf_mean_predict = lambda t: self.lf_model.predict(np.array([t]))[
            0][0]

    def hf_fit(self, hf_X, hf_Y):
        assert self.__lf_mean_predict is not None, "low-fidelity predict function must be given"
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

        a, b = np.min(self.hf_X), np.max(self.hf_X)

        X = np.linspace(a, b, 1000)
        predictions = self.predict_means(X.reshape(-1, 1))

        if (self.lf_Y is None):
            self.lf_X = np.linspace(a, b, 50)
            self.lf_Y = np.array([self.__lf_mean_predict(x) for x in self.lf_X])

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
    def f_low(t): return np.sin(8 * pi * t)
    model = GPDataAugmentation(tau=.01, n=1, input_dims=1)
    model.lf_fit(lf_X=X_train_lf, lf_Y=y_train_lf)
    model.hf_fit(hf_X=X_train_hf, hf_Y=y_train_hf)
    predictions = model.predict_means(X_test)
    mse = mean_squared_error(y_true=y_test, y_pred=predictions)
    print('mean squared error: {}'.format(mse))
    model.plot()


# TODO child classes:
# TODO plot method
#   GP_augmented_data, select better kernel than RBF
#   NARGP
#   MFDGP
# TODO isCheap parameter
