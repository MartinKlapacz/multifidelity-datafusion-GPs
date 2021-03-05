#!./env/bin/python
import numpy as np
import GPy
import matplotlib.pyplot as plt
import time
np.random.seed(1)

import src.data.exampleCurves1D as ex1D
import src.data.exampleCurves2D as ex2D

from src import MultifidelityDataFusion
from src import MethodAssessment
from src import NARGP, GPDF, GPDFC


# if __name__ == "__main__":
#     X_train_hf, X_train_lf, Y_train_lf, f_exact, f_low, X_test, Y_test \
#         = ex1D.get_curve5(num_hf=5, num_lf=100)

#     tau = .01

#     nargp = NARGP(
#         input_dim=1,
#         f_exact=f_exact,
#         f_low=f_low,
#     )

#     gpdf2 = GPDF(
#         name='GPDF2',
#         input_dim=1,
#         tau=tau,
#         num_derivatives=2,
#         f_exact=f_exact,
#         f_low=f_low,
#     )

#     gpdfc2 = GPDFC(
#         name='GPDFC2',
#         input_dim=1,
#         tau=tau,
#         num_derivatives=2,
#         f_exact=f_exact,
#         f_low=f_low,
#     )

#     gpdf4 = GPDF(
#         name='GPDF4',
#         input_dim=1,
#         tau=tau,
#         num_derivatives=4,
#         f_exact=f_exact,
#         f_low=f_low,
#     )

#     gpdfc4 = GPDFC(
#         name='GPDFC4',
#         input_dim=1,
#         tau=tau,
#         num_derivatives=4,
#         f_exact=f_exact,
#         f_low=f_low,
#     )

#     assessment = MethodAssessment([gpdf2, gpdfc2, gpdf4, gpdfc4], X_test=X_test,
#                                   Y_test=Y_test, title="different periodicities")
#     assessment.fit_models(X_train=X_train_hf)
#     assessment.adapt_models(15, plot_mode='e')
#     print(assessment.mses())
#     plt.show()

#     # data driven
#     # input_dim=1, tau=.001, n=1, f_exact=f_exact, adapt_steps=5, lf_X=X_train_lf, lf_Y=y_train_lf, lf_hf_adapt_ratio=1
#     # function driven

if __name__ == '__main__':

    X_train_hf, X_train_lf, Y_train_lf, f_exact, f_low, X_test, Y_test \
        = ex1D.get_discontinuity1(num_hf=5, num_lf=80)
    tau = .01

    nargp = NARGP(
        name='NARGP',
        input_dim=1,
        f_exact=f_exact,
        f_low=f_low,
    )

    gpdf2 = GPDF(
        name='GPDF2',
        input_dim=1,
        tau=tau,
        num_derivatives=2,
        f_exact=f_exact,
        f_low=f_low,
        # add_noise=True,
    )

    gpdfc2 = GPDFC(
        name='GPDFC2',
        input_dim=1,
        tau=tau,
        num_derivatives=2,
        f_exact=f_exact,
        f_low=f_low,
        # add_noise=True,
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

    # models = [gpdf2, gpdfc2, gpdf4, gpdfc4, nargp]
    # names = ['GPDF2', 'GPDFC2', 'GPDF4', 'GPDFC4', 'NARGP']
    models = [gpdfc4]
    names = ['GPDFC4']
    
    sizes = [5, 10, 15, 20, 25]
    num_adapt = 4
    assert len(sizes) == (num_adapt + 1)

    mses = np.zeros((len(models), len(sizes)))

    nruns = 1


    # for i, model in enumerate(models):
        # for j, size in enumerate(sizes):
            # for _ in range(nruns):
                # X_train_hf, X_train_lf, Y_train_lf, f_exact, f_low, X_test, Y_test \
                    # = ex1D.get_curve4(num_hf=size, num_lf=80)

                # model.fit(X_train_hf)
                # mses[i, j] += model.get_mse(X_test, Y_test) / nruns
    
    for i, model in enumerate(models):
        model.fit(X_train_hf)
        mses[i, 0] = model.get_mse(X_test, Y_test)
        for j in range(num_adapt):
            model.adapt(5)
            mses[i, j+1] = model.get_mse(X_test, Y_test)
        model.plot()
        plt.show()
    print(mses)




    for i, row in enumerate(mses):
        plt.plot(sizes, row, '>-', label=names[i])
        plt.xticks(sizes)

    plt.yscale('log')
    plt.xlabel('hf-points', fontsize=16)
    plt.ylabel('mse', fontsize=16)
    # plt.title('Phase Shifted Oscillations', fontsize=20)
    plt.legend()
    plt.show()
