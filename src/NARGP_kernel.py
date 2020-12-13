from GPy.kern import Kern, RBF, Matern32
from GPy.core.parameterization.param import Param
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt


class NARGPKernel(Kern):
    def __init__(self, input_dim: int, n: int, kernClass1: Kern = RBF, kernClass2: Kern = RBF, kernClass3: Kern = RBF,):
        super(NARGPKernel, self).__init__(
            input_dim, np.arange(input_dim), 'NARGPKernel')
        augm_dim = 2*n+1
        standard_entries = np.arange(0, input_dim - augm_dim)
        augm_entries = np.arange(input_dim - augm_dim, input_dim)

        self.kern1 = kernClass1(len(standard_entries), active_dims=standard_entries, ARD=True)
        self.kern2 = kernClass2(len(augm_entries),active_dims=augm_entries, ARD=True)
        self.kern3 = kernClass3(len(standard_entries), active_dims=standard_entries, ARD=True)

        self.kernel = self.kern3 * self.kern2 + self.kern3

        self.variance = Param('variance', self.kernel)
        self.lengthscale = Param('lengtscale', self.kernel)
        # self.add_parameters(self.variance, self.lengthscale)

    def __str__(self):
        return self.kernel.__str__()

    def __add__(self, other):
        return self.kernel.__add__(other)

    def __sub__(self, other):
        return self.kernel.__sub__(other)

    def __mul__(self, other):
        return self.kernel.__mul__(other)

    def __pow__(self, n):
        return self.kernel.__pow(n)

    def K(self, X, X2, ):
        return self.kernel.K(X, X2)

    def Kdiag(self, X, ):
        return self.kernel.Kdiag(X)

    def plot(self, ):
        self.kernel.plot()

    def update_gradients_full(self, dL_dK, X, X2=None):
        # todo: correct implementation
        self.kern1.update_gradients_full(dL_dK, X, X2)
        self.kern2.update_gradients_full(dL_dK, X, X2)
        self.kern3.update_gradients_full(dL_dK, X, X2)