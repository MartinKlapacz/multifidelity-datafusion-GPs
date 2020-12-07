from GPy.kern import Kern, RBF, Matern32
from GPy.core.parameterization.param import Param
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt


class NARPGKernel(Kern):
    def __init__(self, input_dim: int, n: int, kernClass1: Kern = RBF, kernClass2: Kern = RBF, kernClass3: Kern = RBF, variance=1., lengthscale=1., power=1., ):
        super(NARPGKernel, self).__init__(input_dim, np.arange(input_dim), 'NARGPKernel')
        standard_entries = np.arange(0, input_dim)
        augm_length = 2*n+1
        augm_entries = np.arange(input_dim, input_dim + augm_length)

        kern1 = kernClass1(input_dim, active_dims=standard_entries, ARD=True)
        kern3 = kernClass3(input_dim, active_dims=standard_entries, ARD=True)
        kern2 = kernClass2(input_dim=augm_entries, active_dims=augm_entries, ARD=True)

        self.kernel = kern1 * kern2 + kern3
        self.variance = Param('variance', self.kernel)
        self.lengthscale = Param('lengtscale', self.kernel)

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
