import numpy as np
import src.models as models
import test_mfgp_adapt_2d as test_2d
import test_mfgp_adapt_3d as test_3d
import test_mfgp_adapt_4d as test_4d
import src.gpc.chaospy_wrapper as cpw
import chaospy as cp
import src.gpc.mfgp_gpc as mfgp_gpc
import src.models as models
import matplotlib.pyplot as plt


def analytical_mean(a, constant=0):
    if not isinstance(a, list):
        a = [a]
    return np.prod([((1 - np.cos(a_i)) / a_i) for a_i in a]) + constant


def analytical_var(a):
    if not isinstance(a, list):
        a = [a]
    m = analytical_mean(a, constant=0)
    term1 = np.prod([0.5 - (np.sin(2 * a_i) / (4 * a_i)) for a_i in a])
    term2 = m**2
    term3 = 2 * m * np.prod([(np.cos(a_i) - 1) / a_i for a_i in a]) * ((-1)**(len(a)-1))
    return term1 + term2 + term3


def create_data(lf, hf, dim, num_lf=100, num_hf=5, num_test=100):
    X_lf = np.random.uniform(low=0, high=1, size=(num_lf,dim))
    X_hf = np.random.uniform(low=0, high=1, size=(num_hf,dim))
    X_test = np.random.uniform(low=0, high=1, size=(num_test,dim))
    Y_lf, Y_hf = lf(X_lf), hf(X_hf)
    return X_lf, Y_lf, X_hf, Y_hf, X_test


def create_mfgp_obj(dim, lf, hf, X_hf, method='GPDF', add_noise=True):
    model = None
    if method == 'GPDF':
        model = models.GPDF(dim, 0.001, 2, hf, lf, add_noise=add_noise)
    elif method == 'NARGP':
        model = models.NARGP(dim, hf, lf, add_noise=add_noise)
    else:
        model = models.GPDFC(dim, 0.001, 2, hf, lf, add_noise=add_noise)
    model.fit(X_hf)
    return model

def get_a_lf_hf_function(dim):
    if dim == 2:
        return test_2d.a, test_2d.lf_2d, test_2d.hf_2d
    elif dim == 3:
        return test_3d.a, test_3d.lf_3d, test_3d.hf_3d
    elif dim ==4:
        return test_4d.a, test_4d.lf_4d, test_4d.hf_4d
    else:
        print("Wrong input dimension")
        return None


def get_joint_uniform_distribution(dim, lower=0, upper=1):
    if dim == 1:
        return cp.Uniform(lower, upper)
    elif dim == 2:
        return cp.J(cp.Uniform(lower, upper), cp.Uniform(lower, upper))
    elif dim == 3:
        return cp.J(cp.Uniform(lower, upper), cp.Uniform(lower, upper), cp.Uniform(lower, upper))
    elif dim == 4:
        return cp.J(cp.Uniform(lower, upper), cp.Uniform(lower, upper), cp.Uniform(lower, upper), cp.Uniform(lower, upper))
    else:
        print("Wrong input dimension")
        return None


def get_mean_var_mse_mfgpc(dim, lf, hf, X_hf, X_test, method, order, add_noise=True):
    lf_T = lambda x: np.atleast_2d(lf(x)).T
    hf_T = lambda x: np.atleast_2d(hf(x)).T
    Y_test = hf_T(X_test)
    mfgp_obj = create_mfgp_obj(dim, lf_T, hf_T, X_hf, method=method, add_noise=add_noise)
    temp_f = lambda x: mfgp_obj.predict(x)[0]
    distribution = get_joint_uniform_distribution(dim)
    cp_wrapper = cpw.ChaospyWrapper(temp_f, distribution, polynomial_order=order, quadrature_order=order)
    mfgpc = mfgp_gpc.MFGP_GPC(mfgp_obj, cp_wrapper, 13, 5, X_test=X_test, Y_test=Y_test)
    mfgpc.adapt()
    return mfgpc.mean_history, mfgpc.var_history, mfgpc.cost_history, mfgpc.mse_history


def get_order(dim):
    if dim < 4 :
        return 10
    else:
        return 6


def get_gpc_error(dim, start_order=2, end_order=10):
    a, lf, hf = get_a_lf_hf_function(dim)
    distribution  = get_joint_uniform_distribution(dim)
    mean, variance, cost = [], [], []
    for order in range(start_order, end_order+1):
        cp_wrapper1 = cpw.ChaospyWrapper(hf, distribution, polynomial_order=order, quadrature_order=order)
        cp_wrapper1.calculate_coefficients()
        m, v = cp_wrapper1.get_mean_var()
        c = cp_wrapper1.quad_weights.shape[0]
        mean.append(m), variance.append(v), cost.append(c)
    return np.array(mean), np.array(variance), np.array(cost)


if __name__ == '__main__':
    dim, add_noise = 3, True
    a, lf, hf = get_a_lf_hf_function(dim)
    X_lf, Y_lf, X_hf, Y_hf, X_test = create_data(lf, hf, dim)
    actual_mean, actual_variance = analytical_mean(a, constant=5), analytical_var(a)
    print(actual_mean, actual_variance)
    order = get_order(dim)
    # change end order to change the computational cost
    var = False
    if var:
        gpc_mean, gpc_variance, gpc_cost = get_gpc_error(dim, start_order=1, end_order=7)
        nargp_mean, nargp_var, nargp_cost, nargp_mse = get_mean_var_mse_mfgpc(dim, lf, hf, X_hf, X_test, 'NARGP', order)
        gpdf_mean, gpdf_var, gpdf_cost, gpdf_mse = get_mean_var_mse_mfgpc(dim, lf, hf, X_hf, X_test, 'GPDF', order)
        gpdfc_mean, gpdfc_var, gpdfc_cost, gpdfc_mse = get_mean_var_mse_mfgpc(dim, lf, hf, X_hf, X_test, 'GPDFC', order)
        plt.plot(gpc_cost, np.abs((gpc_variance - actual_variance)/actual_variance), label='Direct GPC')
        plt.plot(nargp_cost, np.abs((nargp_var - actual_variance)/actual_variance), label='NARGP')
        plt.plot(gpdf_cost, np.abs((gpdf_var - actual_variance)/actual_variance), label='GPDF')
        plt.plot(nargp_cost, np.abs((gpdfc_var - actual_variance)/actual_variance), label='GPDFC')
        plt.ylabel('Relative error variance', fontsize=16)
    else:
        gpc_mean, gpc_variance, gpc_cost = get_gpc_error(dim, start_order=1, end_order=3)
        nargp_mean, nargp_var, nargp_cost, nargp_mse = get_mean_var_mse_mfgpc(dim, lf, hf, X_hf, X_test, 'NARGP', order)
        gpdf_mean, gpdf_var, gpdf_cost, gpdf_mse = get_mean_var_mse_mfgpc(dim, lf, hf, X_hf, X_test, 'GPDF', order)
        gpdfc_mean, gpdfc_var, gpdfc_cost, gpdfc_mse = get_mean_var_mse_mfgpc(dim, lf, hf, X_hf, X_test, 'GPDFC', order)
        plt.plot(gpc_cost, np.abs((gpc_mean - actual_mean)/actual_mean), label='Direct GPC')
        plt.plot(nargp_cost, np.abs((nargp_mean - actual_mean)/actual_mean), label='NARGP')
        plt.plot(gpdf_cost, np.abs((gpdf_mean - actual_mean)/actual_mean), label='GPDF')
        plt.plot(nargp_cost, np.abs((gpdfc_mean - actual_mean)/actual_mean), label='GPDFC')
        plt.ylabel('Relative error mean', fontsize=16)
    plt.xlabel('Computational Cost', fontsize=16)
    plt.yscale('log')
    plt.legend()
    plt.show() 
