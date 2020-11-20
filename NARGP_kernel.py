from GPy.kern import Kern, RBF, Matern32
from GPy.core.parameterization.param import Param
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt


class NARPGKernel(Kern, ):
    def __init__(self, input_dim: int, kernClass1: Kern, kernClass2: Kern, kernClass3: Kern, variance=1., lengthscale=1., power=1., ):

        active_dims = np.arange(0, input_dim)
        super(NARPGKernel, self).__init__(input_dim, active_dims, 'NARGP')

        self.variance = Param('variance', variance, )
        self.lengthscale = Param('lengtscale', lengthscale, )

        kern1 = kernClass1(input_dim,   active_dims=active_dims, ARD=True)
        kern2 = kernClass2(input_dim=1, active_dims=[input_dim], ARD=True)
        kern3 = kernClass3(input_dim,   active_dims=active_dims, ARD=True)

        self.kernel = kern1 * kern2 + kern3

    def __str__(self):
        return self.kernel.__str__()

    def K(self, X, X2, ):
        self.kernel.K(X, X2)

    def Kdiag(self, X, ):
        self.kernel.Kdiag(X)

    def plot(self, ):
        self.kernel.plot()