#!./env/bin/python
from src.MFDataFusion import MultifidelityDataFusion
from src.abstractMFGP import AbstractMFGP
import numpy as np


class NARGP(Deligator):

    def __init__(self, name: str, input_dim: int, tau: float, f_exact: callable, f_low: callable,
                 lower_bound: np.ndarray = None, upper_bound: np.ndarray = None, lf_X: np.ndarray = None, lf_Y: np.
                 ndarray = None, lf_hf_adapt_ratio: int = 1,):

        self.MFDF = MultifidelityDataFusion(
            name=name, input_dim=input_dim, num_derivatives=0, tau=tau, f_exact=f_exact, lower_bound=lower_bound,
            upper_bound=upper_bound, f_low=f_low, lf_X=lf_X, lf_Y=lf_Y, lf_hf_adapt_ratio=lf_hf_adapt_ratio,
            use_composite_kernel=True)

    def __getattribute__(self, name):
        return super().__getattribute__(name)

    def fit(self, hf_X):
        self.MFDF.fit(hf_X=hf_X)

    def adapt(self, adapt_steps: int, plot_mode: str = None, X_test: np.ndarray = None, Y_test: np.ndarray = None):
        self.MFDF.adapt(adapt_steps, plot_mode, X_test, Y_test)

    def predict(self, X_test: np.ndarray):
        return self.MFDF.predict(X_test=X_test)

    def get_mse(self, X_test: np.ndarray, Y_test: np.ndarray):
        return self.MFDF.get_mse(X_test=X_test, Y_test=Y_test)

    def plot(self):
        self.MFDF.plot()

    def plot_forecast(self):
        self.MFDF.plot_forecast()

    def plot_uncertainties_2D(self):
        self.MFDF.plot_uncertainties_2D(self)


class A(object):

    def __init__(self):
        self.a = 'abc'

    def f(self, x, y):
        return x * y - 20


class B(A):

    def __init__(self):
        self.a = A()

    def __getattribute__(self, name):
        return self.a.__getattribute__(name)



class Delegator(object):

    def wrapper(*args, **kwargs):
        delegation_config = getattr(self, 'DELEGATED_METHODS', None)
        if not isinstance(delegation_config, dict):
            __raise_standard_exception()

        for delegate_object_str, delegated_methods in delegation_config.items():
            if called_method in delegated_methods:
                break
        else:
            __raise_standard_error()

        delegate_object = getattr(self, delegate_object_str, None)

        return getattr(delegate_object, called_method)(*args, **kwargs)


class Parent(Delegator):
    DELEGATED_METHODS = {
        'child': [
            'take_out_the_trash',
            'do_the_dishes'
        ],
        'spouse': [
            'cook_dinner'
        ]
    }

    def __init__(self):
        self.child = Child()
        self.spouse = Spouse()
