import GPy
import numpy as np
import matplotlib.pyplot as plt

from src.abstractGP import AbstractGP
from src.augmentationIterators import EvenAugmentation, BackwardAugmentation
from sklearn.metrics import mean_squared_error
from scipy.optimize import fmin
import time
import sys
import multiprocessing
import concurrent.futures
from scipydirect import minimize


def timer(func):
    def wrapper(*args):
        start = time.time()
        res = func(*args)
        end = time.time()
        print("hf point acquisition duration: {}".format(end - start))
        return res
    return wrapper


class MultifidelityDataFusion(AbstractGP):

    def __init__(self, name: str, input_dim: int, num_derivatives: int, tau: float, f_high: callable,
                 lower_bound: float = None, upper_bound: float = None, f_low: callable = None, lf_X: np.ndarray = None,
                 lf_Y: np.ndarray = None, lf_hf_adapt_ratio: int = 1, use_composite_kernel: bool = True,):

        # model paramters
        self.name = name
        self.input_dim = input_dim
        self.num_derivatives = num_derivatives
        self.tau = tau
        self.f_exact = f_high
        self.f_low = f_low
        self.lf_hf_adapt_ratio = lf_hf_adapt_ratio

        # data bounds
        if lower_bound is None and upper_bound is None:
            self.lower_bound = np.zeros(input_dim)
            self.upper_bound = np.ones(input_dim)
        else:
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

        # augmentation pattern
        self.augm_iterator = EvenAugmentation(self.num_derivatives, dim=input_dim)
        self.optimize_restarts = 10

        self.__initialize_kernel(use_composite_kernel)

        self.__initialize_lf_model(f_low, lf_X, lf_Y, lf_hf_adapt_ratio)

    def __initialize_kernel(self, use_composite_kernel):
        """either use the composite NARGP-kernel or standard RBF"""
        if use_composite_kernel:
            self.kernel = self.NARGP_kernel()
        else:
            new_input_dims = self.input_dim + self.augm_iterator.new_entries_count()
            self.kernel = GPy.kern.RBF(new_input_dims)

    def __initialize_lf_model(self, f_low, lf_X, lf_Y, lf_hf_adapt_ratio):
        """inititialize the low-fidelity model by defining the low-fidelity prediction method self.__lf_mean_predict"""

        # check if the parameters are correctly given
        lf_model_params_are_valid = (f_low is not None) ^ (
            (lf_X is not None) and (lf_Y is not None) and (lf_hf_adapt_ratio is not None))
        assert lf_model_params_are_valid, 'define low-fidelity model either by mean function or by Data'

        self.data_driven_lf_approach = f_low is None
        if self.data_driven_lf_approach:
            # self.__update_input_borders(lf_X)
            self.lf_X = lf_X
            self.lf_Y = lf_Y

            self.lf_model = GPy.models.GPRegression(
                X=lf_X, Y=lf_Y, initialize=True
            )
            self.lf_model.optimize()
            self.__adapt_lf()
            self.__lf_mean_predict = lambda t: self.lf_model.predict(t)[0]
        else:
            self.__lf_mean_predict = f_low

    def fit(self, hf_X):
        """fit the high-fidelity model"""
        assert hf_X.ndim == 2, "invalid input shape"
        assert hf_X.shape[1] == self.input_dim, "invalid input dim"
        self.hf_X = hf_X

        # compute corresponding exact y-values
        self.hf_Y = self.f_exact(self.hf_X)
        assert self.hf_Y.shape == (self.hf_X.shape[0], 1)

        # create the high-fidelity model on augmented X and exact Y
        self.hf_model = GPy.models.GPRegression(
            X=self.__augment_Data(self.hf_X),
            Y=self.hf_Y,
            kernel=self.kernel,
            initialize=True
        )
        self.hf_model.optimize_restarts(num_restarts=6, verbose=False)  # ARD

    def adapt(self, adapt_steps, plot=None, X_test=None, Y_test=None):
        """optimized training of the MFDF model using adapation"""
        assert adapt_steps > 0, 'at least one adapt step necessary'
        self.adapt_steps = adapt_steps

        # different plot modes available
        adapt_modes = {
            'u': lambda: self.__adapt_plot_uncertainties(X_test=X_test, Y_test=Y_test),
            'm': lambda: self.__adapt_plot_means(X_test=X_test, Y_test=Y_test),
            'e': lambda: self.__adapt_plot_error(X_test=X_test, Y_test=Y_test),
            'um': lambda: self.__adapt_plot_combined(X_test=X_test, Y_test=Y_test),
            'mu': lambda: self.__adapt_plot_combined(X_test=X_test, Y_test=Y_test),
            None: lambda: self.__adapt_no_plot(X_test=X_test, Y_test=Y_test),
        }

        assert plot in adapt_modes.keys(), 'invalid plot mode'
        adapt_modes.get(plot)()

    def __adapt_plot_uncertainties(self, X_test=None, Y_test=None, verbose=False):
        assert self.input_dim == 1, "only 2d plotting possible"
        # define input space X
        X = np.linspace(self.lower_bound, self.upper_bound, 200).reshape(-1, 1)
        # prepare subplotting
        subplots_per_row = int(np.ceil(np.sqrt(self.adapt_steps)))
        subplots_per_column = int(np.ceil(self.adapt_steps / subplots_per_row))
        fig, axs = plt.subplots(
            subplots_per_row,
            subplots_per_column,
            sharey='row',
            sharex=True,
            figsize=(20, 10))
        fig.suptitle('Uncertainty development during adaptation ')

        for i in range(self.adapt_steps):
            acquired_x = self.get_input_with_highest_uncertainty()
            _, uncertainties = self.predict(X)
            ax = axs.flatten()[i] if self.adapt_steps > 1 else axs
            ax.axes.xaxis.set_visible(False)
            mse = np.round(self.assess_mse(X_test, Y_test), 4)
            ax.set_title('mse: {}, hf. points: {}'
                         .format(mse, len(self.hf_X)))
            ax.plot(X, uncertainties)
            ax.plot(acquired_x, 0, 'rx')
            self.fit(np.vstack((self.hf_X, acquired_x)))

    def __adapt_plot_means(self, X_test=None, Y_test=None, verbose=False):
        assert self.input_dim == 1, "only 2d plotting possible"
        X = np.linspace(self.lower_bound, self.upper_bound, 200).reshape(-1, 1)
        for i in range(self.adapt_steps):
            acquired_x = self.get_input_with_highest_uncertainty()
            means, _ = self.predict(X)
            plt.plot(X, means, label='step {}'.format(i))
            self.fit(np.vstack((self.hf_X, acquired_x)))
        plt.legend()

    def __adapt_plot_combined(self, X_test=None, Y_test=None, verbose=False):
        assert self.input_dim == 1, "only 2d plotting possible"
        X = np.linspace(self.lower_bound, self.upper_bound, 200).reshape(-1, 1)
        fig, axs = plt.subplots(
            2,
            self.adapt_steps,
            sharey='row',
            sharex=True,
            figsize=(20, 10))

        axs[0][0].set_ylabel('mean curves', size='large')
        axs[1][0].set_ylabel('uncertainty curves', size='large')

        for i in range(self.adapt_steps):
            acquired_x = self.get_input_with_highest_uncertainty()
            means, uncertainty = self.predict(X)
            means = means.flatten()
            uncertainty = uncertainty.flatten()

            mean_ax = axs[0][i]
            mean_ax.set_title('{} hf-points'.format(len(self.hf_X)))
            mean_ax.plot(X, means, 'g')
            mean_ax.plot(X, self.f_low(X), 'r')
            mean_ax.plot(X, self.f_exact(X), 'b')
            mean_ax.plot(self.hf_X, self.hf_Y, 'bx')
            mean_ax.fill_between(X.flatten(),
                                 y1=means - 2 * uncertainty,
                                 y2=means + 2 * uncertainty,
                                 color=(0, 1, 0, .75)
                                 )

            uncertainty_ax = axs[1][i]
            uncertainty_ax.plot(X, uncertainty)
            uncertainty_ax.plot(acquired_x.reshape(-1, 1), 0, 'rx')

            self.fit(np.vstack((self.hf_X, acquired_x)))

    def __adapt_plot_error(self, X_test=None, Y_test=None, yscale='log'):
        assert self.input_dim > 0
        mses = []
        for i in range(self.adapt_steps):
            acquired_x = self.get_input_with_highest_uncertainty()
            self.fit(np.vstack((self.hf_X, acquired_x)))
            mse = self.assess_mse(X_test, Y_test)
            mses.append(mse)
        hf_X_len_before = len(self.hf_X) - self.adapt_steps
        hf_X_len_now = len(self.hf_X)
        plt.title('mean square error')
        plt.xlabel('hf points')
        plt.ylabel('mse')
        plt.yscale(yscale)
        plt.plot(
            np.arange(hf_X_len_before, hf_X_len_now),
            np.array(mses),
            label=self.name
        )
        plt.legend()

    def __adapt_no_plot(self, X_test=None, Y_test=None, verbose=False):
        for i in range(self.adapt_steps):
            acquired_x = self.get_input_with_highest_uncertainty()
            new_hf_X = np.vstack((self.hf_X, acquired_x))
            assert new_hf_X.shape == (len(self.hf_X) + 1, self.input_dim)
            self.fit(new_hf_X)

    def __acquisition_curve(self, x):
        if x.ndim == 1:
            x = x[:, None]
        _, uncertainty = self.predict(x)
        return - uncertainty[:, None]

    def get_input_with_highest_uncertainty(self):
        # negative uncertainty curve
        # def acquisition_curve(x): return - self.predict(x)[1]
        bounds = np.hstack((self.lower_bound[:, None], self.upper_bound[:, None]))
        # maximizing uncertainty is equal to minimizing negative uncertainty curve
        t = time.time()
        res = minimize(self.__acquisition_curve, bounds, maxT=50, )

        print(time.time() - t)
        return res.x

    def __adapt_lf(self):
        X = np.linspace(self.lower_bound, self.upper_bound, 100).reshape(-1, 1)
        for i in range(self.adapt_steps * self.lf_hf_adapt_ratio):
            uncertainties = self.lf_model.predict(X)[1]
            maxIndex = np.argmax(uncertainties)
            new_x = X[maxIndex].reshape(-1, 1)
            new_y = self.lf_model.predict(new_x)[0]

            self.lf_X = np.append(self.lf_X, new_x, axis=0)
            self.lf_Y = np.append(self.lf_Y, new_y, axis=0)

            self.lf_model = GPy.models.GPRegression(
                self.lf_X, self.lf_Y, initialize=True
            )
            self.lf_model.optimize_restarts(
                num_restarts=5,
                optimizer='tnc'
            )

    def predict(self, X_test):
        assert X_test.ndim == 2
        assert X_test.shape[1] == self.input_dim
        X_test = self.__augment_Data(X_test)
        return self.hf_model.predict(X_test)

    def plot(self):
        assert self.input_dim <= 2, 'data must be 1 or 2 dimensional'
        self.__plot()

    def plot_forecast(self, forecast_range=.5):
        self.__plot(exceed_range_by=forecast_range)

    def assess_mse(self, X_test, y_test):
        assert len(X_test) == len(
            y_test), 'number of input values and targets must be equal'
        assert X_test.shape[1] == self.input_dim, 'wrong input value dimension'
        assert y_test.shape[1] == 1, 'target values must be scalars'

        preds, _ = self.predict(X_test)
        mse = mean_squared_error(y_true=y_test, y_pred=preds)
        return mse

    def NARGP_kernel(self, kern_class1=GPy.kern.RBF, kern_class2=GPy.kern.RBF, kern_class3=GPy.kern.RBF):
        std_input_dim = self.input_dim
        std_indezes = np.arange(self.input_dim)

        aug_input_dim = self.augm_iterator.new_entries_count()
        aug_indezes = np.arange(self.input_dim, self.input_dim + aug_input_dim)

        kern1 = kern_class1(aug_input_dim, active_dims=aug_indezes)
        kern2 = kern_class2(std_input_dim, active_dims=std_indezes)
        kern3 = kern_class3(std_input_dim, active_dims=std_indezes)
        return kern1 * kern2 + kern3

    def __plot(self, confidence_inteval_width=2, plot_lf=True, plot_hf=True, plot_pred=True, exceed_range_by=0):
        if (self.input_dim == 1):
            self.__plot1D(confidence_inteval_width, plot_lf, plot_hf, plot_pred, exceed_range_by)
        else:
            self.__plot2D(plot_lf, plot_hf, plot_pred)

    def __plot1D(self, confidence_inteval_width=2, plot_lf=True, plot_hf=True, plot_pred=True, exceed_range_by=0):
        point_density = 1000
        X = np.linspace(self.lower_bound, self.upper_bound * (1 + exceed_range_by),
                        int(point_density * (1 + exceed_range_by))).reshape(-1, 1)
        mean, uncertainty = self.predict(X.reshape(-1, 1))
        mean = mean.flatten()
        uncertainty = uncertainty.flatten()

        if (not self.data_driven_lf_approach):
            self.lf_X = np.linspace(
                self.lower_bound, self.upper_bound, 50).reshape(-1, 1)
            self.lf_Y = self.__lf_mean_predict(self.lf_X)

        lf_color, hf_color, pred_color = 'r', 'b', 'g'

        plt.figure()
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
            plt.plot(X, self.f_exact(X), hf_color,
                     label='f_high', linestyle='dashed')

        if plot_pred:
            # plot prediction
            plt.plot(X, mean, pred_color, label='prediction')
            plt.fill_between(X.flatten(),
                             y1=mean - confidence_inteval_width * uncertainty,
                             y2=mean + confidence_inteval_width * uncertainty,
                             color=(0, 1, 0, .75)
                             )

        plt.legend()
        if self.name:
            plt.title(self.name)

    def __plot2D(self, plot_lf=True, plot_hf=True, plot_pred=True):
        density = 35
        X1 = np.linspace(self.lower_bound[0], self.upper_bound[1], density)
        X2 = np.linspace(self.lower_bound[0], self.upper_bound[1], density)
        X1, X2 = np.meshgrid(X1, X2)
        X = np.array((X1.flatten(), X2.flatten())).T
        preds, _ = self.predict(X)
        lf_y = self.f_low(X)
        hf_y = self.f_exact(X)

        ax = plt.gca(projection='3d')
        if plot_pred:
            ax.scatter(X1, X2, preds)
        if plot_lf:
            ax.scatter(X1, X2, lf_y)
        if plot_hf:
            ax.scatter(X1, X2, hf_y)
        plt.show()


    def __augment_Data(self, X):
        """augment the hf-inputs with corresponding lf-predictions"""
        assert X.shape == (len(X), self.input_dim)

        # number of new entries for each x in X
        new_entries_count = self.augm_iterator.new_entries_count()

        # compute the neighbour positions of each x in X where f_low will be evaluated
        augm_locations = np.array(list(map(lambda x: [x + i * self.tau for i in self.augm_iterator], X)))
        assert augm_locations.shape == (len(X), new_entries_count, self.input_dim)

        # compute the lf-prediction on those neighbour positions
        new_augm_entries = np.array(list(map(self.__lf_mean_predict, augm_locations)))
        assert new_augm_entries.shape == (len(X), new_entries_count, 1)

        # flatten the results of f_low
        new_entries = np.array([entry.flatten() for entry in new_augm_entries])
        assert new_entries.shape == (len(X), new_entries_count)

        # concatenate each x of X with the f_low evaluations at its neighbours
        augmented_X = np.concatenate([X, new_entries], axis=1)
        assert augmented_X.shape == (len(X), new_entries_count + self.input_dim)

        return augmented_X
