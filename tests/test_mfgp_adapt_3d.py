import numpy as np
import src.gpc.chaospy_wrapper as cpw
import chaospy as cp
import src.gpc.mfgp_gpc as mfgp_gpc
import src.models as models
import utils as utils


def hf_3d(param):
    x = np.atleast_2d(param)
    return np.sin(x[:, 0] * 3 * np.pi) * np.sin(x[:, 1] * 2 * np.pi) * np.sin(x[:, 2] * 1 * np.pi) + 5


def lf_3d(param):
    x = np.atleast_2d(param)
    return hf_3d(x) - 0.25 * (np.sin(x[:,0] * np.pi *0.1) + np.sin(x[:,1] * np.pi *0.05) + np.sin(x[:, 2] * 0.15 * np.pi))


lf_3d_T = lambda x: np.atleast_2d(lf_3d(x)).T
hf_3d_T = lambda x: np.atleast_2d(hf_3d(x)).T


def create_mfgp_obj(dim, lf, hf, X_hf):
    # model = models.GPDF(dim, 0.001, 2, hf, lf)
    model = models.NARGP(dim, 0.001, hf, lf)
    model.fit(X_hf)
    return model

if __name__ == '__main__':
    dim = 3
    X_lf, Y_lf, X_hf, Y_hf, X_test = utils.create_data(lf_3d, hf_3d, dim)
    Y_test = hf_3d_T(X_test)
    mfgp_obj = create_mfgp_obj(dim, lf_3d_T, hf_3d_T, X_hf)
    a = [3 * np.pi, 2 * np.pi, 1 * np.pi]
    actual_mean, actual_variance = utils.analytical_mean(a, constant=5), utils.analytical_var(a)
    distribution = cp.J(cp.Uniform(0, 1), cp.Uniform(0, 1), cp.Uniform(0, 1))
    temp_f = lambda x: mfgp_obj.predict(x)[0]
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
    print(mfgpc.cost_history)
    print("MSE", mfgp_obj.get_mse(X_test, Y_test))
    cp_wrapper1 = cpw.ChaospyWrapper(hf_3d, distribution, polynomial_order=9, quadrature_order=9)
    cp_wrapper1.calculate_coefficients()
    mean, variance = cp_wrapper1.get_mean_var()
    print("Chaospy mean", mean)
    print("Chaospy variance", variance)
    relative_error_mean, relative_error_variance = np.abs((mean - actual_mean) / actual_mean), \
                                                  np.abs((variance - actual_variance) / actual_variance)
    print("Error in mean", relative_error_mean)
    print("Error in variance", relative_error_variance)
