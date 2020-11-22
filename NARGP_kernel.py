from GPy.kern import Kern, RBF, Matern32
from GPy.core.parameterization.param import Param
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt


class NARPGKernel(Kern):
    def __init__(self, input_dim: int, kernClass1: Kern, kernClass2: Kern, kernClass3: Kern, variance=1., lengthscale=1., power=1., ):
        assert input_dim >= 2, "input dimension at least 2"

        active_dims = np.arange(0, input_dim)
        super(NARPGKernel, self).__init__(input_dim, active_dims, 'NARGP')

        kern1 = kernClass1(input_dim,   active_dims=active_dims, ARD=True)
        kern2 = kernClass2(input_dim=1, active_dims=[input_dim-1], ARD=True)
        kern3 = kernClass3(input_dim,   active_dims=active_dims, ARD=True)

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


X = np.array([[normal(), normal()] for i in range(5)])

kernel = NARPGKernel(2, RBF, RBF, RBF)
print(kernel.Kdiag(X))

rbf = RBF(2)