import numpy as np
import src.gpc.chaospy_wrapper as cpw
import chaospy as cp
import src.gpc.mfgp_gpc as mfgp_gpc
import src.models as models
import utils as utils
import matplotlib.pyplot as plt

a = [2.2 * np.pi, np.pi]


def hf_2d(param):
    x = np.atleast_2d(param)
    return np.sin(x[:, 0] * a[0]) * np.sin(x[:, 1] * a[1])


def lf_2d(param):
    x = np.atleast_2d(param)
    return hf_2d(x) - 1.2 * (np.sin(x[:, 0] * np.pi * 0.1) + np.sin(x[:, 1] * np.pi * 0.1))


def lf_2d_T(x): return np.atleast_2d(lf_2d(x)).T
def hf_2d_T(x): return np.atleast_2d(hf_2d(x)).T


def create_mfgp_obj(dim, lf, hf, X_hf):
    model = models.GPDF(dim, 0.001, 2, hf, lf)
    model.fit(X_hf)
    return model


if __name__ == '__main__':
    dim = 2
    X_lf, Y_lf, X_hf, Y_hf, X_test = utils.create_data(lf_2d, hf_2d, dim)
    Y_test = hf_2d_T(X_test)
    mfgp_obj = create_mfgp_obj(dim, lf_2d_T, hf_2d_T, X_hf)
    actual_mean, actual_variance = utils.analytical_mean(a), utils.analytical_var(a)
    distribution = cp.J(cp.Uniform(0, 1), cp.Uniform(0, 1))
    def temp_f(x): return mfgp_obj.predict(x)[0]
    cp_wrapper = cpw.ChaospyWrapper(temp_f, distribution, polynomial_order=10, quadrature_order=10)
    print("Analytical Mean", actual_mean)
    print("Analytical Variance", actual_variance)
    cp_wrapper.calculate_coefficients()
    mean, variance = cp_wrapper.get_mean_var()
    print("Chaospy mean", mean)
    print("Chaospy variance", variance)
    mfgpc = mfgp_gpc.MFGP_GPC(mfgp_obj, cp_wrapper, 5, 5)
    mfgpc.adapt()
    print(np.abs(np.array(mfgpc.mean_history) - actual_mean)/actual_mean)
    print(np.abs(np.array(mfgpc.var_history) - actual_variance)/actual_variance)

    plt.plot(mfgpc.cost_history, np.abs(np.array(mfgpc.mean_history) - actual_mean)/actual_mean, 'v-')
    plt.yscale('log')
    plt.ylabel('relative error', fontsize=16)
    plt.xlabel('hf points', fontsize=16)
    plt.title('mean estimation history', fontsize=20)
    plt.show()
    plt.plot(mfgpc.cost_history, np.abs(np.array(mfgpc.var_history) - actual_variance)/actual_variance, 'v-')
    plt.yscale('log')
    plt.ylabel('relative error', fontsize=16)
    plt.xlabel('hf points', fontsize=16)
    plt.title('variance estimation history', fontsize=20)
    plt.show()

    # print(mfgpc.cost_history)
    print("MSE", mfgp_obj.get_mse(X_test, Y_test))

# [[2.0774e+00]
#  [3.3235e-02]
#  [3.7437e-03]
#  [1.2300e-03]
#  [1.7799e-04]
#  [3.5887e-04]]
# [[6.5727e-01]
#  [3.6365e-03]
#  [4.8415e-05]
#  [6.0635e-05]
#  [1.8925e-05]
#  [8.7902e-06]]
# [5, 10, 15, 20, 25, 30]