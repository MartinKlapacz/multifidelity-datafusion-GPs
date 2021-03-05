import GPy
import numpy as np
import matplotlib.pyplot as plt
from DIRECT import solve

from .abstractMFGP import AbstractMFGP
from .adaptation_maximizers import DIRECTLMaximizer, AbstractMaximizer, ScipyDirectMaximizer
from .augm_iterators import *
from sklearn.metrics import mean_squared_error
from scipy.optimize import fmin


class MultifidelityDataFusion(AbstractMFGP):
    """This is a machine learning model meant for regression tasks in a multifielity setting, 
    where a scarce/precise (high-fidelity) and an abundant/imprecise (low-fidelity) data set
    are available.

    :param name: name of the model which will be used in plots
    :type name: str

    :param input_dim: dimension of the input vectors in the data sets
    :type input_dim: int

    :param num_derivatives: number of implicitly included derivatives, 
    for given n: derivatives 0, ..., n are computed
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

    :param use_composite_kernel: if True use composite NARGP kernel, otherwise RBF kernel, defaults to True
    :type use_composite_kernel: bool, optional
    """

    def __init__(self, name: str, input_dim: int, num_derivatives: int, tau: float, f_exact: callable,
                 lower_bound: np.ndarray = None, upper_bound: float = None, f_low: callable = None, lf_X: np.ndarray = None,
                 lf_Y: np.ndarray = None, lf_hf_adapt_ratio: int = 1, use_composite_kernel: bool = True,
                 adapt_maximizer: AbstractMaximizer = ScipyDirectMaximizer(),
                 eps: float = 1e-8, iteratorClass: AbstractAugmIterator = EvenAugmentation, add_noise: bool = False):

        super().__init__(name=name, input_dim=input_dim, num_derivatives=num_derivatives,
                         tau=tau, f_exact=f_exact, lower_bound=lower_bound, upper_bound=upper_bound, f_low=f_low,
                         lf_X=lf_X, lf_Y=lf_Y, lf_hf_adapt_ratio=lf_hf_adapt_ratio,
                         use_composite_kernel=use_composite_kernel, adapt_maximizer=adapt_maximizer, eps=eps,
                         iteratorClass=iteratorClass)

        self.initialize_kernel(use_composite_kernel)

        self.initialize_lf_level(f_low, lf_X, lf_Y)

        self.add_noise = add_noise

    def fit(self, hf_X):
        """fits the model by fitting its high-fidelity model with a augmented
        version of the input high-fidelity training data 

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
        self.ARD(self.hf_model, 6)

    def adapt(self, adapt_steps: int, plot_mode: str = None, X_test: np.ndarray = None, Y_test: np.ndarray = None, eps: float = 1e-8):
        """optimizes the hf-model by acquiring new hf-training data points, which at each step,
        reduce the uncertainty of the model the most. The new point will be the one, whose corresponding 
        prediction value whould come with the highest uncertainty.

        :param adapt_steps: number of new high-fidelity data points
        :type adapt_steps: int
        :param plot_mode: possible modes are 'm', 'u', 'e', and 'mu', defaults to None
        :type plot_mode: string, optional
        :param X_test: test input data, defaults to None
        :type X_test: np.ndarray, optional
        :param Y_test: test target data, defaults to None
        :type Y_test: np.ndarray, optional
        :param eps: if (during the adaptation) the max. uncertainty becomes smaller than eps, 
        stop the adaptation process as the model is sufficiently fitted
        :type eps: float, optional
        """

        self.adapt_steps = adapt_steps
        self.X_test = X_test
        self.Y_test = Y_test
        self.eps = eps

        if self.data_driven_lf_approach:
            self.__adapt_lf()

        # there are different plot modes available
        adapt_mode_dict = {
            'u': lambda: self.adapt_and_plot(plot_uncertainties=True),
            'm': lambda: self.adapt_and_plot(plot_means=True),
            'e': lambda: self.adapt_and_plot(plot_error=True),
            'um': lambda: self.adapt_and_plot(plot_means=True, plot_uncertainties=True),
            'mu': lambda: self.adapt_and_plot(plot_means=True, plot_uncertainties=True),
            None: lambda: self.adapt_and_plot(),
        }

        assert plot_mode in adapt_mode_dict.keys(), "Invalid plot mode. Select one of these: {}" % adapt_mode_dict.keys()
        adapt_mode_dict.get(plot_mode)()

    def predict(self, X_test):
        """for an array of input vectors computes the corresponding 
        target values

        :param X_test: input vectors
        :type X_test: np.ndarray
        :return: target values per input vector
        :rtype: np.ndarray
        """

        assert X_test.ndim == 2
        assert X_test.shape[1] == self.input_dim
        X_test = self.__augment_Data(X_test)
        if self.add_noise:
            self.hf_model.likelihood.variance = 1e-6
        return self.hf_model.predict(X_test)

    def get_mse(self, X_test, Y_test):
        """compute the mean square error the given test data

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

    def __augment_Data(self, X):
        """augments high-fidelity inputs with corresponding low-fidelity predictions.
        The augmentation pattern is determined by self.augm_iterator

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
        new_augm_entries = np.array(list(map(self.f_low, augm_locations)))
        assert new_augm_entries.shape == (len(X), new_entries_count, 1)

        # flatten the results of f_low
        new_entries = np.array([entry.flatten() for entry in new_augm_entries])
        assert new_entries.shape == (len(X), new_entries_count)

        # concatenate each x of X with the f_low evaluations at its neighbours
        augmented_X = np.concatenate([X, new_entries], axis=1)
        assert augmented_X.shape == (len(X), new_entries_count + self.input_dim)

        return augmented_X
