#!./env/bin/python
import numpy as np
import GPy
import matplotlib.pyplot as plt
import time

import src.data.exampleCurves1D as ex1D
import src.data.exampleCurves2D as ex2D

from src import MultifidelityDataFusion
from src import MethodAssessment
from src import NARGP, GPDF, GPDFC


if __name__ == "__main__":
    X_train_hf, X_train_lf, Y_train_lf, f_exact, f_low, X_test, Y_test \
        = ex1D.get_curve4(num_hf=9, num_lf=80)

    tau = .01

    nargp = NARGP(
        input_dim=1,
        f_exact=f_exact,
        f_low=f_low,
    )

    nargp.fit(X_train_hf)
    nargp.adapt(2, 'mu', X_test, Y_test)
    plt.show()


# mses1 = {
#     'NARGP':  1.7439106517828003e-08,
#     'GPDFC1': 3.289036321241657e-08,
#     'GPDFC2': 2.7566617459388876e-08
# }
# mses2 = {
#     'NARGP':  2.61785104812192,
#     'GPDFC1': 3.615431439205672e-07,
#     'GPDFC2': 3.6153595106526963e-07
# }
