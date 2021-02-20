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
        = ex1D.get_curve1(num_hf=20, num_lf=100)

    tau = .01

    gpdfc = GPDFC(
        input_dim=1,
        tau=.01,
        num_derivatives=2,
        f_exact=f_exact,
        f_low=f_low,
    )

    gpdfc.fit(X_train_hf)
    print(gpdfc.plot_lengthscale_hyperparams())
    plt.show()