import numpy as np
import abc
import GPy
import matplotlib.pyplot as plt
from DIRECT import solve
from .adaptation_maximizers import DIRECT1_maximizer, AbstractMaximizer


class AbstractMFGP(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self, name: str, input_dim: int, num_derivatives: int, tau: float, f_exact: callable,
                 lower_bound: np.ndarray, upper_bound: float, f_low: callable, lf_X: np.ndarray, lf_Y: np.ndarray,
                 lf_hf_adapt_ratio: int, use_composite_kernel: bool, adapt_maximizer: AbstractMaximizer):

        super().__init__()
        self.name = name
        self.input_dim = input_dim
        self.num_derivatives = num_derivatives
        self.tau = tau
        self.f_exact = f_exact
        self.f_low = f_low
        self.lf_hf_adapt_ratio = lf_hf_adapt_ratio
        self.adapt_maximizer = adapt_maximizer

        # data bounds
        if lower_bound is None and upper_bound is None:
            self.lower_bound = np.zeros(input_dim)
            self.upper_bound = np.ones(input_dim)
        else:
            self.lower_bound = lower_bound
            self.upper_bound = upper_bound

    @abc.abstractmethod
    def fit(self, hf_X):
        pass

    @abc.abstractmethod
    def adapt(self, adapt_steps, plot_mode, X_test, Y_test):
        pass

    @abc.abstractmethod
    def predict(self, X_test):
        pass

    @abc.abstractmethod
    def get_mse(self, X_test, Y_test):
        pass

    def initialize_kernel(self, use_composite_kernel: bool):
        """initializes kernel of hf-model, either use composite NARGP kernel or standard RBF
        :param use_composite_kernel: use composite NARGP kernel
        :type use_composite_kernel: bool
        """
        if use_composite_kernel:
            self.kernel = self.get_NARGP_kernel()
        else:
            new_input_dims = self.input_dim + self.augm_iterator.new_entries_count()
            self.kernel = GPy.kern.RBF(new_input_dims)

    def get_NARGP_kernel(self, kern_class1=GPy.kern.RBF, kern_class2=GPy.kern.RBF, kern_class3=GPy.kern.RBF):
        """build composite NARGP kernel with proper dimension and kernel classes
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

    def initialize_lf_level(self, f_low: callable = None, lf_X: np.ndarray = None, lf_Y: np.ndarray = None):
        """initialize low-fidelity level by python function or by trained GP model,
        pass either a lf prediction function or lf training data
        :param f_low: low fidelity prediction function
        :type f_low: callable
        :param lf_X: low fidelity input vectors
        :type lf_X: np.ndarray
        :param lf_Y: low fidelity input target values
        :type lf_Y: np.ndarray
        """
        # check if the parameters are correctly given
        lf_model_params_are_valid = (f_low is not None) ^ (
            (lf_X is not None) and (lf_Y is not None) and (self.lf_hf_adapt_ratio is not None))
        assert lf_model_params_are_valid, 'define low-fidelity model either by predicition function or by data'
        self.data_driven_lf_approach = f_low is None
        if self.data_driven_lf_approach:
            self.lf_X = lf_X
            self.lf_Y = lf_Y
            self.lf_model = GPy.models.GPRegression(
                X=lf_X, Y=lf_Y, initialize=True
            )
            self.lf_model.optimize()
            self.f_low = lambda t: self.lf_model.predict(t)[0]
        else:
            self.f_low = f_low

    def adapt_lf(self):
        """optimizes the hf-model by acquiring additional hf-training points for training"""

        assert hasattr(self, 'lf_model'), "lf-model not initialized"
        for i in range(self.adapt_steps * self.lf_hf_adapt_ratio):
            acquired_x = self.get_input_with_highest_uncertainty(self.lf_model)
            acquired_y = self.lf_model.predict(acquired_x[None])[0][0]

            self.lf_X = np.vstack((self.lf_X, acquired_x))
            self.lf_Y = np.vstack((self.lf_Y, acquired_y))

            self.lf_model = GPy.models.GPRegression(
                self.lf_X, self.lf_Y, initialize=True
            )
            self.__ARD(self.lf_model, 6)

    def get_input_with_highest_uncertainty(self, model):
        """get input from input domain whose prediction comes with the highest uncertainty"""
        assert hasattr(model, 'predict')

        x, fopt = self.adapt_maximizer.maximize(self.predict, self.lower_bound, self.upper_bound)
        return x, fopt

    def ARD(self, model, num_restarts):
        model[".*Gaussian_noise"] = model.Y.var()*0.01
        model[".*Gaussian_noise"].fix()
        model.optimize(max_iters=500)
        model[".*Gaussian_noise"].unfix()
        model[".*Gaussian_noise"].constrain_positive()
        model.optimize_restarts(num_restarts, optimizer="bfgs",  max_iters=1000, verbose=False)

    def plot(self):
        """plot low-fidelity mean, high-fidelity mean and exact mean curve"""

        assert self.input_dim in [1, 2], 'data must be 1 or 2 dimensional'
        self.__plot()

    def plot_forecast(self, forecast_range=.5):
        """plot low-fidelity mean, high-fidelity mean and exact mean curve with
        extended x-axis

        :param forecast_range: x-axis range in the figure is given by 
        (1 + forecast_range) * (upper_bound - lower_bound), defaults to .5
        :type forecast_range: float, optional
        """
        self.__plot(exceed_range_by=forecast_range)

    def plot_uncertainties_2D(self):
        """3D plot the of the uncertainty plane"""

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
        ax.plt_surface(X1, X2, )
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
        confidence_inteval_width = 2
        X = np.linspace(self.lower_bound, self.upper_bound * (1 + exceed_range_by),
                        int(point_density * (1 + exceed_range_by))).reshape(-1, 1)
        mean, uncertainty = self.predict(X.reshape(-1, 1))
        mean = mean.flatten()
        uncertainty = uncertainty.flatten()

        if (not self.data_driven_lf_approach):
            self.lf_X = np.linspace(
                self.lower_bound, self.upper_bound, 50).reshape(-1, 1)
            self.lf_Y = self.f_low(self.lf_X)

        lf_color, hf_color, pred_color = 'r', 'b', 'g'

        plt.figure()
        if plot_lf:
            # plot low fidelity
            plt.plot(self.lf_X, self.lf_Y, lf_color +
                     'x', label='low-fidelity')
            plt.plot(X, self.f_low(X), lf_color,
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

    def adapt_and_plot(
            self, plot_means: bool = False, plot_uncertainties: bool = False, plot_error: bool = False, eps: float = 1e-8):
        """model adaptation and plotting to illustrate the process of optimization

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
            acquired_x, fopt = self.get_input_with_highest_uncertainty(self)
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
            if np.abs(fopt) < eps:
                # stop after condition
                # print massage
                # self.adapt_steps = i

                # print("Iteration stopped after {} iterations!" % i
                #      + " minimum uncertainty reached: {}" % fopt)
                break

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
