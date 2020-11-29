import GPy
import numpy as np
import matplotlib.pyplot as plt
from abstractGP import AbstractGP

def augmentIter(n):
    i = 0
    sign = -1
    while i < n or sign == 1:
        if i == 0: yield 0
        if sign == 1:
            sign = -1
        else:
            sign = 1
            i += 1
        yield sign * i


class DataAugmentationGP(AbstractGP):

    def __init__(self, tau: float, n: int, input_dims: int, f_low: callable=None, lf_X: np.ndarray=None, lf_Y: np.ndarray=None):
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
        self.hf_model = None

        lf_model_params_are_valid = (f_low is not None) ^ ((lf_X is not None) and (lf_Y is not None))
        assert lf_model_params_are_valid, 'define low-fidelity model either by mean function or by Data'

        self.data_driven_lf_approach = f_low is None
        if self.data_driven_lf_approach:
            self.lf_X, self.lf_Y = lf_X, lf_Y
            self.lf_model = GPy.models.GPRegression(
                X=lf_X, Y=lf_Y, initialize=True
            )
            self.lf_model.optimize()
            self.__lf_mean_predict = lambda t: self.lf_model.predict(t)[0]
        else:
            self.__lf_mean_predict = f_low

    def fit(self, hf_X, hf_Y):
        assert self.__lf_mean_predict is not None, "low-fidelity predict function must be given"
        assert hf_X.ndim == 2
        self.hf_X, self.hf_Y = hf_X, hf_Y
        augmented_hf_X = self.__augment_Data(hf_X)
        self.hf_model = GPy.models.GPRegression(
            X=augmented_hf_X, Y=hf_Y, initialize=True
        )
        self.hf_model.optimize()  # ARD

    def predict(self, X_test):
        assert X_test.ndim == 2
        assert X_test.shape[1] == self.input_dims
        X_test = self.__augment_Data(X_test)
        return self.hf_model.predict(X_test)

    def predict_means(self, X_test):
        return self.predict(X_test)[0]

    def plot(self):
        assert self.input_dims == 1, '2d plots need one-dimensional data'
        assert self.hf_model is not None, 'model is not fitted yet'

        a, b = np.min(self.hf_X), np.max(self.hf_X)

        X = np.linspace(a, b, 500)
        predictions = self.predict_means(X.reshape(-1, 1))

        if (not self.data_driven_lf_approach):
            self.lf_X = np.linspace(a, b, 50)
            self.lf_Y = self.__lf_mean_predict(self.lf_X)

        plt.plot(self.lf_X, self.lf_Y, 'ro', label='low-fidelity')
        plt.plot(self.hf_X, self.hf_Y, 'bo', label='high-fidelity')
        plt.plot(X, predictions, label='prediction', linestyle='dashed')
        plt.legend()
        plt.show()

    def __augment_Data(self, X):
        assert isinstance(X, np.ndarray), 'input must be an array'
        assert len(X) > 0, 'input must be non-empty'
        augmented_X = np.concatenate([
            self.__lf_mean_predict(X + i * self.tau) for i in augmentIter(self.n)
        ], axis=1)
        return np.concatenate([X, augmented_X], axis=1)

# TODO child classes:
# TODO plot method
#   GP_augmented_data, select better kernel than RBF
#   NARGP
#   MFDGP
# TODO isCheap parameter