import GPy
import numpy as np
import matplotlib.pyplot as plt

from src.abstractGP import AbstractGP
from src.NARGP_kernel import NARGPKernel
from src.augmentationIterators import augmentIter
from sklearn.metrics import mean_squared_error


class DataAugmentationGP(AbstractGP):

    def __init__(self, tau: float, n: int, input_dim: int, f_high: callable, f_low: callable = None, lf_X: np.ndarray = None, lf_Y: np.ndarray = None, step_count: int = 0):
        '''
        input: tau
            distance to neighbour points used in taylor expansion
        input n: 
            number of derivatives which will be included when training the high-fidelity model,
            adds 2*n+1 dimensions to the high-fidelity training data
        input input_dim:
            dimensionality of the input data
        input f_low:
            closed form of a low-fidelity prediction function, 
            if not provided, call self.lf_fit() to train a low-fidelity GP which will be used for low-fidelity predictions instead
        '''
        self.tau = tau
        self.n = n
        self.input_dim = input_dim
        self.__f_high_real = f_high
        self.f_low = f_low
        self.a = self.b = None

        lf_model_params_are_valid = (f_low is not None) ^ (
            (lf_X is not None) and (lf_Y is not None))
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

    def fit(self, hf_X):
        hf_X = hf_X.reshape(-1, 1)
        # high fidelity data is as precise as ground truth data
        self.hf_X, self.hf_Y = hf_X, self.__f_high_real(hf_X)
        # augment input data before prediction
        augmented_hf_X = self.__augment_Data(hf_X)

        kernel = NARGPKernel(input_dim=augmented_hf_X.shape[1], n=self.n)

        self.hf_model = GPy.models.GPRegression(
            X=augmented_hf_X, Y=self.hf_Y, kernel=None, initialize=True
        )
        self.hf_model.optimize()  # ARD

    def adapt_hf(self, num_steps):
        # do the same with low fidelity, with more steps (ration * numsteps) ration given in __init__
        assert self.hf_model is not None
        for i in range(num_steps):
            # find x with highest variance
            acquired_x = self.get_x_with_highest_uncertainty()
            self.fit(np.append(self.hf_X, acquired_x))

    def predict(self, X_test):
        assert X_test.ndim == 2
        assert X_test.shape[1] == self.input_dim
        X_test = self.__augment_Data(X_test)
        return self.hf_model.predict(X_test)

    def predict_means(self, X_test):
        return self.predict(X_test)[0]

    def predict_variance(self, X_test):
        return self.predict(X_test)[1]

    def plot(self):
        assert self.input_dim == 1, '2d plots need one-dimensional data'
        assert self.f_high is not None, 'model is not fitted yet'
        self.__plot()

    def plot_forecast(self, forecast_range=.5):
        self.__plot(exceed_range_by=forecast_range)

    def assess_mse(self, X_test, y_test):
        predictions = self.predict_means(X_test)
        mse = mean_squared_error(y_true=y_test, y_pred=predictions)
        print('mean squared error: {}'.format(mse))
        return mse

    def __plot(self, confidence_inteval_width=2, plot_lf=True, plot_hf=True, plot_pred=True, exceed_range_by=0):
        self.a, self.b = np.min(self.hf_X), np.max(self.hf_X)
        point_density = 500
        X = np.linspace(self.a, self.b * (1 + exceed_range_by),
                        int(point_density * (1 + exceed_range_by)))
        pred_mean, pred_variance = self.predict(X.reshape(-1, 1))
        pred_mean = pred_mean.flatten()
        pred_variance = pred_variance.flatten()

        if (not self.data_driven_lf_approach):
            self.lf_X = np.linspace(self.a, self.b, 50)
            self.lf_Y = self.__lf_mean_predict(self.lf_X)

        lf_color, hf_color, pred_color = 'r', 'b', 'g'

        if plot_lf:
            # plot low fidelity
            plt.plot(self.lf_X, self.lf_Y, lf_color +
                     'x', label='low-fidelity')
            plt.plot(X, self.__lf_mean_predict(X), lf_color,
                     label='f_low', linestyle='dashed')

        if plot_hf:
            # plot high fidelity
            plt.plot(self.hf_X, self.hf_Y, hf_color +
                     'x', label='high-fidelity')
            plt.plot(X, self.__f_high_real(X), hf_color,
                     label='f_high', linestyle='dashed')

        if plot_pred:
            # plot prediction
            plt.plot(X, pred_mean, pred_color, label='prediction')
            plt.fill_between(X,
                             y1=pred_mean - confidence_inteval_width * pred_variance,
                             y2=pred_mean + confidence_inteval_width * pred_variance,
                             color=(0, 1, 0, .75)
                             )

        plt.legend()
        plt.show()

    def __augment_Data(self, X):
        assert isinstance(X, np.ndarray), 'input must be an array'
        assert len(X) > 0, 'input must be non-empty'
        new_entries = np.concatenate([
            self.__lf_mean_predict(X + i * self.tau) for i in augmentIter(self.n)
        ], axis=1)
        return np.concatenate([X, new_entries], axis=1)

    def get_x_with_highest_uncertainty(self, precision: int = 200):
        if self.a is None:
            self.a = np.min(self.hf_X)
        if self.b is None:
            self.b = np.max(self.hf_X)
        X = np.linspace(self.a, self.b, precision).reshape(-1, 1)
        uncertainties = self.predict_variance(X)
        index_with_highest_uncertainty = np.argmax(uncertainties)
        return X[index_with_highest_uncertainty]

#   GP_augmented_data, select better kernel than RBF
#   NARGP
#   MFDGP

# complete adapt (just for high) (look in GPY, bayesian optimization toolbox)
# complete adapt (for low)
# combine both
# check results
# describe the process
