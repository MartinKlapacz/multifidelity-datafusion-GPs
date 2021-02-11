import numpy as np
import src.gpc.chaospy_wrapper as cpw
import chaospy as cp
import src.gpc.mfgp_gpc as mfgp_gpc
import src.models as models
import utils as utils
import time


def hf_4d(param):
    x = np.atleast_2d(param)
    return np.sin(x[:, 0] * np.pi) * np.sin(x[:, 1] * np.pi) * np.sin(x[:, 2] * np.pi) \
           * np.sin(x[:, 3] * np.pi) + 5


def lf_4d(param):
    x = np.atleast_2d(param)
    return hf_4d(x) - 0.25 * (np.sin(x[:,0] * np.pi * 0.1) + np.sin(x[:,1] * np.pi *0.05)
                              + np.sin(x[:, 2] * 0.15 * np.pi) + np.sin(x[:, 3] * 0.2 * np.pi))


lf_4d_T = lambda x: np.atleast_2d(lf_4d(x)).T
hf_4d_T = lambda x: np.atleast_2d(hf_4d(x)).T


def create_mfgp_obj(dim, lf, hf, X_hf):
    # model = models.GPDF(dim, 0.001, 2, hf, lf)
    model = models.NARGP(dim, 0.001, hf, lf)
    model.fit(X_hf)
    return model

if __name__ == '__main__':
    dim = 4
    X_lf, Y_lf, X_hf, Y_hf, X_test = utils.create_data(lf_4d, hf_4d, dim)
    Y_test = hf_4d_T(X_test)
    mfgp_obj = create_mfgp_obj(dim, lf_4d_T, hf_4d_T, X_hf)
    a = [np.pi, np.pi, np.pi, np.pi]
    actual_mean, actual_variance = utils.analytical_mean(a, constant=5), utils.analytical_var(a)
    distribution = cp.J(cp.Uniform(0, 1), cp.Uniform(0, 1), cp.Uniform(0, 1), cp.Uniform(0, 1))
    temp_f = lambda x: mfgp_obj.predict(x)[0]
    cp_wrapper = cpw.ChaospyWrapper(temp_f, distribution, polynomial_order=6, quadrature_order=6)
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
    start = time.time()
    cp_wrapper1 = cpw.ChaospyWrapper(hf_4d, distribution, polynomial_order=4, quadrature_order=4)
    cp_wrapper1.calculate_coefficients()
    end = time.time()
    print("Time taken to calculate the coefficients ", end - start)
    start = time.time()
    mean, variance = cp_wrapper1.get_mean_var()
    end = time.time()
    print("Time to calculate statistical moments ", end - start)
    print("Chaospy mean", mean)
    print("Chaospy variance", variance)
    relative_error_mean, relative_error_variance = np.abs((mean - actual_mean) / actual_mean), \
                                                  np.abs((variance - actual_variance) / actual_variance)
    print("Error in mean", relative_error_mean)
    print("Error in variance", relative_error_variance)


'''
Observation: We set the polynomial order and quadrature order to 10, and set separate timers on calculate_coefficients
and get_mean_var. We observe that that calculation of coefficients takes 170 seconds and calculation of mean and 
variance takes around 210 seconds. Chaospy is doing something strange. It should not take this much of time. Specially, 
after calculation of coefficients, the calculation of mean and variance is just extraction of coefficients.
TODO (only for Ravi): Integrate your own code to calculate statistical moments
'''