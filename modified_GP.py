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

    def __init__(self, lff: callable, tau: float, n: int, input_dims: int):
        '''
        input: lff
            low fidelity function
        input: tau
            distance to neighbour points used in taylor expansion
        input n: 
            number of lags in the augmentation process (2*n + 1)
        '''
        self.lff = lff
        self.tau = tau
        self.n = n
        self.input_dims = input_dims
        self.hf_model = None

    def fit(self, X, Y):
        assert X.ndim == 2
        assert Y.ndim == 2
        assert X.shape[1] == self.input_dims, 'invalid input dimension'
        assert len(X) == len(Y), 'not match lengths'

        augmented_X = self.__augment_vector_list(X)
        # kernel = NARPGKernel(self.input_dims)
        self.hf_model = GPy.models.GPRegression(
            augmented_X, Y, initialize=True
        )
        self.hf_model.optimize()  # ARD

    def predict(self, x: float):
        assert x.ndim == 1, 'prediction input must be vector'
        assert len(x) == self.input_dims, 'invalid input dimension'

        augmented_x = self.__augment_vector(x)
        return self.hf_model.predict(np.array([augmented_x]))

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
        scalar = x[0]
        x = np.append(x, self.lff(scalar))
        # if n > 0 include lagged values
        for i in range(1, self.n+1):
            x = np.append(x, self.lff(scalar + i * self.tau))
            x = np.append(x, self.lff(scalar - i * self.tau))
        return x

    def __augment_vector_list(self, X):
        assert isinstance(X, np.ndarray), 'input must be an array'
        assert len(X) > 0, 'input must be non-empty'

        augmented_X = np.array([self.__augment_vector(x) for x in X])
        print(augmented_X.shape)
        print(self.input_dims + 2 * self.n + 1)
        assert augmented_X.shape[1] == self.input_dims + 2 * self.n + 1
        return augmented_X


if __name__ == "__main__":
    # TODO use training data from gaussian process summer school 2020
    X_train, y_train, X_test, y_test = generate_Mauna_Loa_data(reduce_by=2)

    # init low fidelity model
    lf_model = GPy.models.GPRegression(X=X_train, Y=y_train)
    lf_model.optimize()

    def lff(x): 
        model_input = np.array([[x]])
        model_output = lf_model.predict(model_input)
        y = model_output[0][0][0]

        assert type(y) == np.float64
        return y
    
    hf_model = GPDataAugmentation(lff, tau=.5, n=2, input_dims=1)
    hf_model.fit(X=X_train, Y=y_train)

# TODO abstract class, multiple implementations of GP Methods inheriting of abstract class
# TODO child classes:
#   GP_augmented_data, select better kernel than RBF
#   NARGP
#   MFDGP
# TODO isCheap parameter
# TODO store plots
