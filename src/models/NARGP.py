#!./env/bin/python
from src.MFDataFusion import MultifidelityDataFusion
from src.abstractMFGP import AbstractMFGP
import numpy as np


class NARGP:
    """Nonlinear autoregressive multi-fidelity GP model,
    expects high-fidelity data to train the high-fidelity its model and low-fidelity data to train its
    low-fidelity model. Augments its high-fidelity data only with low-fidelity predictions.
    Uses composite NARGP kernel with ARD weights.
    """

    def __init__(self, input_dim: int, tau: float, f_exact: callable, f_low: callable, name: str = 'NARGP',
                 lower_bound: np.ndarray = None, upper_bound: np.ndarray = None, lf_X: np.ndarray = None, lf_Y: np.
                 ndarray = None, lf_hf_adapt_ratio: int = 1,):

        self.MFDF = MultifidelityDataFusion(
            name=name, input_dim=input_dim, num_derivatives=0, tau=tau, f_exact=f_exact, lower_bound=lower_bound,
            upper_bound=upper_bound, f_low=f_low, lf_X=lf_X, lf_Y=lf_Y, lf_hf_adapt_ratio=lf_hf_adapt_ratio,
            use_composite_kernel=True)

        self.name = self.MFDF.name
        self.input_dim = self.MFDF.input_dim
        self.num_derivatives = self.MFDF.num_derivatives
        self.tau = self.MFDF.tau
        self.f_exact = self.MFDF.f_exact
        self.f_low = self.MFDF.f_low
        self.lf_hf_adapt_ratio = self.MFDF.lf_hf_adapt_ratio
        self.adapt_maximizer = self.MFDF.adapt_maximizer

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

    def plot_forecast(self, forecast_range: float):
        self.MFDF.plot_forecast(forecast_range)

    def plot_uncertainties_2D(self):
        self.MFDF.plot_uncertainties_2D(self)
