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

def timer(func):
    def wrapper(*args):
        start = time.time()
        res = func(*args)
        end = time.time()
        print("hf point acquisition duration: {}".format(end - start))
        return res
    return wrapper

class MultifidelityDataFusion(AbstractGP):

    def __init__(self, tau: float, n: int, input_dim: int, f_high: callable, adapt_steps: int = 0, f_low: callable = None, lf_X: np.ndarray = None, lf_Y: np.ndarray = None, lf_hf_adapt_ratio: int = 1,):
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
        self.adapt_steps = adapt_steps
        self.lf_hf_adapt_ratio = lf_hf_adapt_ratio
        self.a = self.b = None
        self.augm_iterator = BackwardAugmentation(self.n, dim=input_dim)
        self.acquired_X = []

        lf_model_params_are_valid = (f_low is not None) ^ (
            (lf_X is not None) and (lf_Y is not None) and (lf_hf_adapt_ratio is not None))
        assert lf_model_params_are_valid, 'define low-fidelity model either by mean function or by Data'

        self.data_driven_lf_approach = f_low is None

        if self.data_driven_lf_approach:
            self.__update_input_borders(lf_X)
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
        self.hf_X = hf_X
        if self.hf_X.ndim == 1:
            self.hf_X = hf_X.reshape(-1,1)
        self.__update_input_borders(hf_X)
        # high fidelity data is as precise as ground truth data

        # TODO
        self.hf_Y = self.__f_high_real(self.hf_X)
        # augment input data before prediction
        augmented_hf_X = self.__augment_Data(self.hf_X)

        self.hf_model = GPy.models.GPRegression(
            X=augmented_hf_X,
            Y=self.hf_Y,
            kernel=self.NARGP_kernel(),
            initialize=True
        )
        self.hf_model.optimize_restarts(num_restarts=6, verbose=False)  # ARD

    def adapt(self, a=None, b=None, plot=None, X_test=None, Y_test=None, verbose=False):
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b
        assert self.adapt_steps > 0
        if plot == 'uncertainty':
            assert self.input_dim == 1
            self.__adapt_plot_uncertainties(
                X_test=X_test, Y_test=Y_test, verbose=verbose)
        elif plot == 'mean':
            assert self.input_dim == 1
            self.__adapt_plot_means(
                X_test=X_test, Y_test=Y_test, verbose=verbose)
        elif plot is None:
            assert self.input_dim > 0
            self.__adapt_no_plot(verbose=verbose)
        else:
            raise Exception(
                'invalid plot mode, use mean, uncertainty or False')

    def __adapt_plot_uncertainties(self, X_test=None, Y_test=None, verbose=False):
        X = np.linspace(self.a, self.b, 200).reshape(-1, 1)
        # prepare subplotting
        subplots_per_row = int(np.ceil(np.sqrt(self.adapt_steps)))
        subplots_per_column = int(np.ceil(self.adapt_steps / subplots_per_row))
        fig, axs = plt.subplots(
            subplots_per_row,
            subplots_per_column,
            sharey='row',
            sharex=True,
            figsize=(20, 10))
        fig.suptitle(
            'Uncertainty development during the adaptation process')
        log_mses = []

        # axs_flat = axs.flatten()
        for i in range(self.adapt_steps):
            acquired_x = self.get_input_with_highest_uncertainty()
            if verbose:
                print('new x acquired: {}'.format(acquired_x))
            _, uncertainties = self.predict(X)
            # todo: for steps = 1, flatten() will fail
            ax = axs.flatten()[i]
            ax.axes.xaxis.set_visible(False)
            log_mse = self.assess_log_mse(X_test, Y_test)
            log_mses.append(log_mse)
            ax.set_title(
                'log mse: {}, high-f. points: {}'.format(log_mse, len(self.hf_X)))
            ax.plot(X, uncertainties)
            ax.plot(acquired_x.reshape(-1, 1), 0, 'rx')
            self.fit(np.append(self.hf_X, acquired_x))

        # plot log_mse development during adapt process
        plt.figure(2)
        plt.title('logarithmic mean square error')
        plt.xlabel('hf points')
        plt.ylabel('log mse')
        hf_X_len_before = len(self.hf_X) - self.adapt_steps
        hf_X_len_now = len(self.hf_X)
        plt.plot(
            np.arange(hf_X_len_before, hf_X_len_now),
            np.array(log_mses)
        )

    def __adapt_plot_means(self, X_test=None, Y_test=None, verbose=False):
        X = np.linspace(self.a, self.b, 200).reshape(-1, 1)
        for i in range(self.adapt_steps):
            acquired_x = self.get_input_with_highest_uncertainty()
            if verbose:
                print('new x acquired: {}'.format(acquired_x))
            means, _ = self.predict(X)
            plt.plot(X, means, label='step {}'.format(i))
            self.fit(np.append(self.hf_X, acquired_x))
        plt.legend()

    def __adapt_no_plot(self, verbose=False):
        for i in range(self.adapt_steps):
            acquired_x = self.get_input_with_highest_uncertainty()
            if verbose:
                print('new x acquired: {}'.format(acquired_x))
            self.fit(np.append(self.hf_X, acquired_x))

    def __acquisition_curve(self, x):
        X = x.reshape(-1, 1)
        uncertainty = self.predict(X)[1]
        return 1 / uncertainty

    @timer
    def get_input_with_highest_uncertainty(self, restarts: int = 20):
        best_xopt = 0
        best_fopt = sys.maxsize
        random_vector = np.random.uniform(size=(restarts, self.input_dim))
        start_positions = self.a + random_vector * (self.b - self.a)

        # TODO: parallelize
        for start in start_positions:
            xopt, fopt, _, _, allvecs = fmin(
                self.__acquisition_curve, start, full_output=True, disp=False)
            if fopt < best_fopt and self.a < xopt and xopt < self.b:
                best_fopt = fopt
                best_xopt = xopt
        return best_xopt

    # def parallel_stuff(self, restarts: int = 20):
    #     best_xopt = 0
    #     best_fopt = sys.maxsize
    #     random_vector = np.random.uniform(size=(restarts, self.input_dim))
    #     start_positions = self.a + random_vector * (self.b - self.a)

    #     cpu_count = multiprocessing.cpu_count()
    #     threads = [thread for thread in range(cpu_count)]

    def __adapt_lf(self):
        X = np.linspace(self.a, self.b, 100).reshape(-1, 1)
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

    def predict_means(self, X_test):
        means, _ = self.predict(X_test)
        return means

    def predict_variance(self, X_test):
        _, uncertainties = self.predict(X_test)
        return uncertainties

    def plot(self):
        assert self.input_dim == 1, 'data must be 2 dimensional in order to be plotted'
        self.__plot()

    def plot_forecast(self, forecast_range=.5):
        self.__plot(exceed_range_by=forecast_range)

    def assess_log_mse(self, X_test, y_test):
        assert X_test.shape[1] == self.input_dim
        assert y_test.shape[1] == 1
        predictions = self.predict_means(X_test)
        mse = mean_squared_error(y_true=y_test, y_pred=predictions)
        log_mse = np.log2(mse)
        return np.round(log_mse, 4)

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
        point_density = 500
        X = np.linspace(self.a, self.b * (1 + exceed_range_by),
                        int(point_density * (1 + exceed_range_by))).reshape(-1, 1)
        pred_mean, pred_variance = self.predict(X.reshape(-1, 1))
        pred_mean = pred_mean.flatten()
        pred_variance = pred_variance.flatten()

        if (not self.data_driven_lf_approach):
            self.lf_X = np.linspace(self.a, self.b, 50).reshape(-1, 1)
            self.lf_Y = self.__lf_mean_predict(self.lf_X)

        lf_color, hf_color, pred_color = 'r', 'b', 'g'

        plt.figure(3)
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
            plt.fill_between(X.flatten(),
                             y1=pred_mean - confidence_inteval_width * pred_variance,
                             y2=pred_mean + confidence_inteval_width * pred_variance,
                             color=(0, 1, 0, .75)
                             )

        plt.legend()

    def __augment_Data(self, X):
        n = len(X)
        new_entries_count = self.augm_iterator.new_entries_count()

        assert X.shape == (len(X), self.input_dim)

        augm_locations = np.array(list(map(lambda x: [x + i * self.tau for i in self.augm_iterator], X)))
        
        assert augm_locations.shape == (len(X), new_entries_count, self.input_dim)
        
        new_augm_entries = self.__lf_mean_predict(augm_locations)
        
        assert new_augm_entries.shape == (len(X), new_entries_count, 1)
        
        new_entries = np.array([entry.flatten() for entry in new_augm_entries])

    
        assert new_entries.shape == (len(X), new_entries_count)

        augmented_X = np.concatenate([X, new_entries], axis=1)

        assert augmented_X.shape == (len(X), new_entries_count + 1)
        
        return augmented_X

    def __update_input_borders(self, X: np.ndarray):
        if self.a == None and self.b == None:
            self.a = np.min(X, axis=0)
            self.b = np.max(X, axis=0)
        else:
            self.a = np.min([self.a, np.min(X, axis=0)], axis=0)
            self.b = np.max([self.b, np.max(X, axis=0)], axis=0)
#   GP_augmented_data, select better kernel than RBF
#   NARGP
#   MFDGP

# complete adapt (just for high) (look in GPY, bayesian optimization toolbox)
# complete adapt (for low)
# combine both
# check results
# describe the process
