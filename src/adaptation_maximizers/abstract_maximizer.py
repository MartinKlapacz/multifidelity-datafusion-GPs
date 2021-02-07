from abc import ABCMeta, abstractmethod
import numpy as np


class AbstractMaximizer(metaclass=ABCMeta):
    """Abstract wrapper class for the uncertainty maximization in the adaptation 
    process.
    """
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def maximize(self, model_predict: callable, lower_bound: np.ndarray, upper_bound: np.ndarray):
        """ computes and returns the global maximizer of the model's uncertainty using a 
        specific algorithm.

        :param uncertainty_curve: prediction function of the model which uses,
        return a tuple (means, uncertainties)
        this optimizer
        :type uncertainty_curve: callable
        :param lower_bound: array of minimum entries for the result
        :type lower_bound: np.ndarray
        :param upper_bound: array of maximium entries for the result
        :type upper_bound: np.ndarray
        :return: returns an array that maximizes the model_predict function
        :rtype: np.ndarray
        """
