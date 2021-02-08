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
        = ex1D.get_curve3(num_hf=5, num_lf=100)

    tau = .01

    gpdf2 = GPDF(
        name='GPDF2',
        input_dim=1,
        tau=tau,
        num_derivatives=2,
        f_exact=f_exact,
        f_low=f_low,
    )

    gpdfc2 = GPDFC(
        name='GPDFC2',
        input_dim=1,
        tau=tau,
        num_derivatives=2,
        f_exact=f_exact,
        f_low=f_low,
    )

    gpdf4 = GPDF(
        name='GPDF4',
        input_dim=1,
        tau=tau,
        num_derivatives=4,
        f_exact=f_exact,
        f_low=f_low,
    )

    gpdfc4 = GPDFC(
        name='GPDFC4',
        input_dim=1,
        tau=tau,
        num_derivatives=4,
        f_exact=f_exact,
        f_low=f_low,
    )



    assessment = MethodAssessment([gpdf2, gpdfc2, gpdf4, gpdfc4], X_test=X_test, Y_test=Y_test, title="phase shifted oscillations")
    assessment.fit_models(X_train=X_train_hf)
    assessment.adapt_models(20, plot_mode='e')
    print(assessment.mses())
    plt.show()

    # data driven
    # input_dim=1, tau=.001, n=1, f_exact=f_exact, adapt_steps=5, lf_X=X_train_lf, lf_Y=y_train_lf, lf_hf_adapt_ratio=1
    # function driven
