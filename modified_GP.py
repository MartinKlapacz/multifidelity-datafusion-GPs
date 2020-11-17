import GPy
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def generate_random_training_data(a, b):
    X = np.linspace(a, b, 10).reshape(-1, 1)
    Y = np.array([np.round(np.sin(i), 2) for i in X]).reshape(-1, 1)
    return X, Y


class Modified_GP_Regression:

    def __init__(self, lff: callable, delta_t: float, derivative_bitmap: [bool]):
        '''
        input: lff
            low fidelity function
        input: delta_t
            distance to neighbour points used in taylor expansion
        input: derivate_bitmap
            describes included derivatives, if i-th entry is true, i-th derivative is included
        '''
        self.lff = lff
        self.delta_t = delta_t
        self.derivative_bitmap = derivative_bitmap

    def fit(self, X, Y):
        augmented_X = self.__augment(X)
        self.hf_model = GPy.models.GPRegression(
            augmented_X, Y, initialize=True
        )
        self.hf_model.optimize()

    def predict(self, x: float):
        x = np.append(x, self.lff(x))
        return self.hf_model.predict(np.array([x]))

    def __augment(self, X):
        """ append the low fidelity prediction to each vector in X """
        augmented_X = np.array([np.append(x, self.lff(x)) for x in X])
        return augmented_X


    def __numeric_derivative(self, f: callable, x: np.array, n: int):
        """ implementing the formula for the nth-derivate stencil """
        sum = 0
        for k in range(n):
            sum = + (-1)**(k + n) * binom(n, k) * f(x+k*self.delta_t)[0]
        return sum / self.delta_t**n


if __name__ == "__main__":

    f = lambda x: 2 * np.sin(x) + 0.01 * x * x
    a = 0
    b = 30
    n = 25

    X = np.linspace(a, b, n).reshape(-1, 1)
    Y = np.array([f(x) for x in X])

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # init low fidelity model
    lf_model = GPy.models.GPRegression(X=X_train, Y=y_train)
    lf_model.optimize()

    print("lf mean squarred error: {}" % mean_squared_error(y_true=y_test, y_pred=lf_model.predict(X_test)[0]))

    def lff(x): return lf_model.predict(np.array([x]))[0]

    # init high fidelity model
    modified_GP = Modified_GP_Regression(lff, delta_t=0.1, derivative_bitmap=[True])
    modified_GP.fit(X_train, y_train)
    print(modified_GP.predict(X_test[0]))