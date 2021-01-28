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
from DIRECT import solve

class MultifidelityDataFusion(AbstractGP):
    """multifidelity Data fusion model 

    :param name: name which will be used in plots
    :type name: str

    :param input_dim: input vector dimension
    :type input_dim: int

    :param num_derivatives: number of implicitly included derivatives
    :type num_derivatives: int

    :param tau: distance to the augmentation neighbors, used for data augmentation
    :type tau: float

    :param f_high: exact high-fidelity function
    :type f_high: callable

    :param lower_bound: array of minimun values for each entry, defaults to None
    :type lower_bound: np.ndarray, optional

    :param upper_bound: array of maximum values for each entry, defaults to None
    :type upper_bound: np.ndarray, optional

    :param f_low: exact low-fidelity function, defaults to None
    :type f_low: callable, optional

    :param lf_X: training data for the low-fidelity model, defaults to None
    :type lf_X: np.ndarray, optional

    :param lf_Y: training data target values for the low-fidelity model, defaults to None
    :type lf_Y: np.ndarray, optional

    :param lf_hf_adapt_ratio: number of lf-adapt-steps = #hf-adapt-steps * lf_hf_adapt_ratio, defaults to 1
    :type lf_hf_adapt_ratio: int, optional

    :param use_composite_kernel: if True NARGP composite kernel used, otherwise RBF kernel, defaults to True
    :type use_composite_kernel: bool, optional
    """

    def __init__(self, name: str, input_dim: int, num_derivatives: int, tau: float, f_high: callable,
                 lower_bound: np.ndarray = None, upper_bound: float = None, f_low: callable = None, lf_X: np.ndarray = None,
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

        # augmentation pattern defined by iterator
        self.augm_iterator = EvenAugmentation(self.num_derivatives, dim=input_dim)

        self.__initialize_kernel(use_composite_kernel)

        self.__initialize_lf_model(f_low, lf_X, lf_Y)

    def __initialize_kernel(self, use_composite_kernel: bool):
        """initializes kernel of hf-model, either use the composite NARGP-kernel or standard RBF

        :param use_composite_kernel: use composite NARGP kernel
        :type use_composite_kernel: bool
        """

        if use_composite_kernel:
            self.kernel = self.get_NARGP_kernel()
        else:
            new_input_dims = self.input_dim + self.augm_iterator.new_entries_count()
            self.kernel = GPy.kern.RBF(new_input_dims)

    def get_NARGP_kernel(self, kern_class1=GPy.kern.RBF, kern_class2=GPy.kern.RBF, kern_class3=GPy.kern.RBF):
        """build composite NARGP kernel

        :param kern_class1: first kernel class, defaults to GPy.kern.RBF
        :type kern_class1: GPy.kern.src.kernel_slice_operations.KernCallsViaSlicerMeta, optional
        :param kern_class2: second kernel class defaults to GPy.kern.RBF
        :type kern_class2: GPy.kern.src.kernel_slice_operations.KernCallsViaSlicerMeta, optional
        :param kern_class3: third kernel class, defaults to GPy.kern.RBF
        :type kern_class3: GPy.kern.Kern, optional
        :return: composite NARGP kernel
        :rtype: GPy.kern.Kern
        """

        std_input_dim = self.input_dim
        std_indezes = np.arange(self.input_dim)

        aug_input_dim = self.augm_iterator.new_entries_count()
        aug_indezes = np.arange(self.input_dim, self.input_dim + aug_input_dim)

        kern1 = kern_class1(aug_input_dim, active_dims=aug_indezes)
        kern2 = kern_class2(std_input_dim, active_dims=std_indezes)
        kern3 = kern_class3(std_input_dim, active_dims=std_indezes)
        return kern1 * kern2 + kern3

    def __initialize_lf_model(self, f_low: callable, lf_X: np.ndarray, lf_Y: np.ndarray):
        """initialize low-fidelity model by python function or by training data

        :param f_low: low fidelity prediction function
        :type f_low: callable
        :param lf_X: low fidelity input vectors
        :type lf_X: np.ndarray
        :param lf_Y: low fidelity input target values
        :type lf_Y: np.ndarray
        """

        # check if the parameters are correctly given
        lf_model_params_are_valid = (f_low is not None) ^ (
            (lf_X is not None) and (lf_Y is not None) and (lf_hf_adapt_ratio is not None))
        assert lf_model_params_are_valid, 'define low-fidelity model either by predicition function or by data'

        self.data_driven_lf_approach = f_low is None
        if self.data_driven_lf_approach:
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

    # workflow

    def fit(self, hf_X):
        """fits the high fidelity model

        :param hf_X: training input vectors
        :type hf_X: np.ndarray
        """

        assert hf_X.ndim == 2, "invalid input shape"
        assert hf_X.shape[1] == self.input_dim, "invalid input dim"
        # save current high-fidelity data (used later in adaptation)
        self.hf_X = hf_X

        # compute corresponding exact y-values
        self.hf_Y = self.f_exact(self.hf_X)
        assert self.hf_Y.shape == (self.hf_X.shape[0], 1)

        # create the high-fidelity model with augmented X and exact Y
        self.hf_model = GPy.models.GPRegression(
            X=self.__augment_Data(self.hf_X),
            Y=self.hf_Y,
            kernel=self.kernel,
            initialize=True
        )
        # ARD steps
        num_restarts = 6
        self.hf_model[".*Gaussian_noise"] = self.hf_model.Y.var()*0.01
        self.hf_model[".*Gaussian_noise"].fix()
        self.hf_model.optimize(max_iters = 500)
        self.hf_model[".*Gaussian_noise"].unfix()
        self.hf_model[".*Gaussian_noise"].constrain_positive()
        self.hf_model.optimize_restarts(num_restarts, optimizer = "bfgs",  max_iters = 1000, verbose=False)

    def adapt(self, adapt_steps:int, plot_mode:str=None, X_test:np.ndarray=None, Y_test:np.ndarray=None):
        """optimize the hf-model by acquiring additional hf-training points for training

        :param adapt_steps: number of new high-fidelity data points
        :type adapt_steps: int
        :param plot_mode: possible modes are 'm', 'u', 'e', and 'mu', defaults to None
        :type plot_mode: string, optional
        :param X_test: [description], defaults to None
        :type X_test: np.ndarray, optional
        :param Y_test: [description], defaults to None
        :type Y_test: np.ndarray, optional
        """

        self.adapt_steps = adapt_steps
        self.X_test = X_test
        self.Y_test = Y_test

        # there are different plot modes available
        adapt_mode_dict = {
            'u':  lambda: self.__adapt_and_plot(plot_uncertainties=True),
            'm':  lambda: self.__adapt_and_plot(plot_means=True),
            'e':  lambda: self.__adapt_and_plot(plot_error=True),
            'um': lambda: self.__adapt_and_plot(plot_means=True, plot_uncertainties=True),
            'mu': lambda: self.__adapt_and_plot(plot_means=True, plot_uncertainties=True),
            None: lambda: self.__adapt_and_plot(),
        }

        assert plot_mode in adapt_mode_dict.keys(), "Invalid plot mode. Select one of these: {}" % adapt_mode_dict.keys()
        adapt_mode_dict.get(plot_mode)()

    def __adapt_lf(self):
        """optimize the hf-model by acquiring additional hf-training points for training"""
        
        # TODO: update
        for i in range(self.adapt_steps * self.lf_hf_adapt_ratio):


            self.lf_X = np.vstack((self.lf_X, None))
            self.lf_Y = np.vstack((self.lf_Y, None))

            self.lf_model = GPy.models.GPRegression(
                self.lf_X, self.lf_Y, initialize=True
            )
            self.lf_model.optimize_restarts(
                num_restarts=5,
                optimizer='tnc'
            )

    def predict(self, X_test):
        """predict function

        :param X_test: input vectors
        :type X_test: np.ndarray
        :return: target values per input vector
        :rtype: np.ndarray
        """

        assert X_test.ndim == 2
        assert X_test.shape[1] == self.input_dim
        X_test = self.__augment_Data(X_test)
        return self.hf_model.predict(X_test)

    def get_mse(self, X_test, Y_test):
        """compute mean square error

        :param X_test: test input vectors
        :type X_test: np.ndarray
        :param Y_test: test target vectors
        :type Y_test: np.ndarray
        :return: mean square error
        :rtype: float
        """

        assert len(X_test) == len(Y_test), 'unequal number of X and y values'
        assert X_test.shape[1] == self.input_dim, 'wrong input value dimension'
        assert Y_test.shape[1] == 1, 'target values must be scalars'

        preds, _ = self.predict(X_test)
        mse = mean_squared_error(y_true=Y_test, y_pred=preds)
        return mse

    def __get_input_with_highest_uncertainty(self):
        """get input from input domain whose prediction comes with the highest uncertainty"""

        def acquisition_curve(x, dummy):
            # DIRECT.solve() calls this function with x and dummy value
            _, uncertainty = self.predict(x[None])
            return - uncertainty[:, None]
        # DIRECT minimisation optimizer
        x, _, _ = solve(acquisition_curve, self.lower_bound, self.upper_bound, maxT=50, algmethod=1)
        return x

    def __augment_Data(self, X):
        """augment high-fidelity inputs with corresponding low-fidelity predictions

        :param X: high-fidelity input vectors
        :type X: np.ndarray
        :return: return augmented high-fidelity input vectors
        :rtype: np.ndarray
        """

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

    # adaptation
    def __adapt_and_plot(self, plot_means: bool=False, plot_uncertainties: bool=False, plot_error: bool=False):
        """optimize the model by adaptation of new high-fidelity points, plot the process

        :param plot_means: plot mean curves, defaults to False
        :type plot_means: bool, optional
        :param plot_uncertainties: plot uncertainty curves, defaults to False
        :type plot_uncertainties: bool, optional
        :param plot_error: plot error curve, defaults to False
        :type plot_error: bool, optional
        """

        X = np.linspace(self.lower_bound, self.upper_bound, 200)
        mses = []

        plot_combined = plot_means and plot_uncertainties
        single_plot = plot_means ^ plot_error and not plot_combined
        # subplot sizes
        if plot_combined:
            nrows = 2
            ncols = self.adapt_steps
        elif single_plot:
            nrows = 1 
            ncols = 1
        elif plot_uncertainties:
            nrows = int(np.ceil(np.sqrt(self.adapt_steps)))
            ncols = int(np.ceil(self.adapt_steps / nrows))

        if plot_means or plot_uncertainties:
            _, axs = plt.subplots(
                nrows,
                ncols,
                sharey='row',
                sharex=True,
                figsize=(20, 10))
        
        if plot_combined:
            axs[0][0].set_ylabel('mean curves', size='large')
            axs[1][0].set_ylabel('uncertainty curves', size='large')

        # adaptation loop
        for i in range(self.adapt_steps):
            acquired_x = self.__get_input_with_highest_uncertainty()
            means, uncertainties = self.predict(X)
            new_hf_X = np.vstack((self.hf_X, acquired_x))
            
            if plot_combined:
                means, uncertainties = means.flatten(), uncertainties.flatten()
                # means row
                mean_ax = axs[0][i]
                mean_ax.set_title('{} hf-points'.format(len(self.hf_X)))
                mean_ax.plot(X, means, 'g')
                mean_ax.plot(X, self.f_low(X), 'r')
                mean_ax.plot(X, self.f_exact(X), 'b')
                mean_ax.plot(self.hf_X, self.hf_Y, 'bx')
                mean_ax.fill_between(X.flatten(),
                                    y1=means - 2 * uncertainties,
                                    y2=means + 2 * uncertainties,
                                    color=(0, 1, 0, .75)
                                    )
                # uncertainty row
                uncertainty_ax = axs[1][i]
                uncertainty_ax.plot(X, uncertainties)
                uncertainty_ax.plot(acquired_x.reshape(-1, 1), 0, 'rx')
            elif plot_uncertainties:
                ax = axs.flatten()[i] if self.adapt_steps > 1 else axs
                ax.axes.xaxis.set_visible(False)
                mse = np.round(self.get_mse(self.X_test, self.Y_test), 4)
                ax.set_title('mse: {}, hf. points: {}'.format(mse, len(self.hf_X)))
                ax.plot(X, uncertainties)
                ax.plot(acquired_x, 0, 'rx')
            elif plot_means:
                axs.plot(X, means, label='step {}'.format(i))
                plt.legend()
            elif plot_error:
                mse = self.get_mse(self.X_test, self.Y_test)
                mses.append(mse)

            self.fit(new_hf_X)

        if plot_error:
            hf_X_len_before = len(self.hf_X) - self.adapt_steps
            hf_X_len_now = len(self.hf_X)
            plt.title('mean square error')
            plt.xlabel('hf points')
            plt.ylabel('mse')
            plt.yscale('log')
            plt.plot(
                np.arange(hf_X_len_before, hf_X_len_now),
                np.array(mses),
                label=self.name
            )
            plt.legend()
        elif single_plot:
            Y = means if plot_means else uncertainties
            plt.plot(X, Y, label='step {}'.format(i))
            plt.legend()

    def plot(self):
        """plot low-fidelity mean, high-fidelity mean and exact mean curve"""
        
        assert self.input_dim in [1, 2], 'data must be 1 or 2 dimensional'
        self.__plot()

    def plot_forecast(self, forecast_range=.5):
        """plot low-fidelity mean, high-fidelity mean and exact mean curve with 

        :param forecast_range: (1 + forecast_range) * (upper_bound - lower_bound) x-axis length, defaults to .5
        :type forecast_range: float, optional
        """
        self.__plot(exceed_range_by=forecast_range)

    def plot_uncertainties_2D(self):
        """3d plot of the uncertainty for 2D input"""

        assert self.input_dim == 2, 'method only callable for 2 dim'
        density = 60
        X1 = np.linspace(self.lower_bound[0], self.upper_bound[1], density)
        X2 = np.linspace(self.lower_bound[0], self.upper_bound[1], density)
        X1, X2 = np.meshgrid(X1, X2)
        X = np.array((X1.flatten(), X2.flatten())).T
        _, uncertainties = self.predict(X)
        ax = plt.gca(projection='3d')
        ax.scatter(X1, X2, uncertainties)
        ax.scatter(self.hf_X[:, 0], self.hf_X[:, 1], 0)
        plt.show()

    def __plot(self, plot_lf=True, plot_hf=True, plot_pred=True, exceed_range_by=0):
        """plot for input_dim=1 or input_dim=2

        :param plot_lf: plot low-fidelity curve, defaults to True
        :type plot_lf: bool, optional
        :param plot_hf: plot high-fidelity curve, defaults to True
        :type plot_hf: bool, optional
        :param plot_pred: plot prediction curve, defaults to True
        :type plot_pred: bool, optional
        :param exceed_range_by: extend plotting domain, defaults to 0
        :type exceed_range_by: float, optional
        """

        if (self.input_dim == 1):
            self.__plot1D(plot_lf, plot_hf, plot_pred, exceed_range_by)
        else:
            self.__plot2D(plot_lf, plot_hf, plot_pred)

    def __plot1D(self, plot_lf=True, plot_hf=True, plot_pred=True, exceed_range_by=0):
        """plot model with input_dim = 1

        :param plot_lf: plot low-fidelity curve, defaults to True
        :type plot_lf: bool, optional
        :param plot_hf: plot high-fidelity curve, defaults to True
        :type plot_hf: bool, optional
        :param plot_pred: plot prediction curve, defaults to True
        :type plot_pred: bool, optional
        :param exceed_range_by: extend plotting domain, defaults to 0
        :type exceed_range_by: float, optional
        """

        point_density = 1000
        confidence_inteval_width=2
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
                             y1=mean - 2 * uncertainty,
                             y2=mean + 2 * uncertainty,
                             color=(0, 1, 0, .75)
                             )

        plt.legend()
        if self.name:
            plt.title(self.name)

    def __plot2D(self, plot_lf=True, plot_hf=True, plot_pred=True):
        """plot model with input_dim = 2

        :param plot_lf: plot low-fidelity curve, defaults to True
        :type plot_lf: bool, optional
        :param plot_hf: plot high-fidelity curve, defaults to True
        :type plot_hf: bool, optional
        :param plot_pred: plot prediction curve, defaults to True
        :type plot_pred: bool, optional
        :param exceed_range_by: extend plotting domain, defaults to 0
        :type exceed_range_by: float, optional
        """

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