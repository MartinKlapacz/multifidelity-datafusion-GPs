import numpy as np


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
