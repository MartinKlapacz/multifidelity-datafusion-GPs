#!./env/bin/python
from src.MFDataFusion import MultifidelityDataFusion
from src.abstractMFGP import AbstractMFGP
from src.MFDataFusion import MultifidelityDataFusion
import numpy as np
import matplotlib.pyplot as plt


class GPDFC(MultifidelityDataFusion):
    """Gaussian Process with Data Fusion and composite kernel, 
    expects high-fidelity data to train its high-fidelity model and low-fidelity data to train its
    low-fidelity model. Augments high-fidelity data with low-fidelity predictions and implicit derivatives.
    Uses composite NARGP kernel with ARD weigts.
    """

    def __init__(self, input_dim: int, tau: float, num_derivatives: int, f_exact: callable, f_low: callable,
                 name: str = 'GPDFC', lower_bound: np.ndarray = None, upper_bound: np.ndarray = None, lf_X: np.ndarray = None,
                 lf_Y: np.ndarray = None, lf_hf_adapt_ratio: int = 1, eps: float = 1e-8, add_noise: bool = False):

        super().__init__(name=name, input_dim=input_dim, num_derivatives=num_derivatives, tau=tau, f_exact=f_exact,
                         lower_bound=lower_bound, upper_bound=upper_bound, f_low=f_low, lf_X=lf_X, lf_Y=lf_Y,
                         lf_hf_adapt_ratio=lf_hf_adapt_ratio, use_composite_kernel=True, eps=eps, add_noise=add_noise)


    def plot_lengthscale_hyperparams(self):
        kern = self.kernel.to_dict()
        lengthscale1, l1 = kern.get("parts").get(1).get("lengthscale")[0], 'l1'
        lengthscale2, l2 = kern.get("parts").get(0).get("parts").get(0).get("lengthscale")[0], 'l2'
        lengthscale3, l3 = kern.get("parts").get(0).get("parts").get(1).get("lengthscale")[0], 'l3'
        l = [l1, l2, l3]
        x = [i for i, _ in enumerate(l)]
        plt.bar(x, [lengthscale1, lengthscale2, lengthscale3])
        plt.xticks(x, l)
        return lengthscale1, lengthscale2, lengthscale3
