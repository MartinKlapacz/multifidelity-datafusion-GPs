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

mses = []
for i in range(5):
    X_train_hf, X_train_lf, Y_train_lf, f_exact, f_low, X_test, Y_test \
        = ex2D.get_curve2(num_hf=5, num_lf=80)

    tau = .01

    model = GPDFC(
        input_dim=2,
        name='adapted data',
        tau=tau,
        num_derivatives=1,
        f_exact=f_exact,
        f_low=f_low,
    )
    model.fit(X_train_hf)
    model.adapt(20)
    mses.append(model.get_mse(X_test, Y_test))

print(mses)
print(sum(mses) / len(mses))



if __name__ == "__main__x":
    X_train_hf, X_train_lf, Y_train_lf, f_exact, f_low, X_test, Y_test \
        = ex2D.get_curve2(num_hf=5, num_lf=80)

    tau = .01

    model1 = NARGP(
        input_dim=2,
        name='adapted data',
        f_exact=f_exact,
        f_low=f_low,
    )

    model2 = GPDF(
        input_dim=2,
        name='adapted data',
        tau=tau,
        num_derivatives=1,
        f_exact=f_exact,
        f_low=f_low,
    )

    model3 = GPDFC(
        input_dim=2,
        name='adapted data',
        tau=tau,
        num_derivatives=1,
        f_exact=f_exact,
        f_low=f_low,
    )

    model4 = GPDFC(
        input_dim=2,
        name='adapted data',
        tau=tau,
        num_derivatives=2,
        f_exact=f_exact,
        f_low=f_low,
    )

    models = [model1, model2, model3, model4]

    for j, model in enumerate(models):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        model.fit(X_train_hf)
        model.adapt(20, 'e', X_test=X_test, Y_test=Y_test)
        ran = range(5, 26, 5)
        mses = []
        for i in ran:
            mses.append(0)
            for _ in range(5):
                X_train_hf, X_train_lf, Y_train_lf, f_exact, f_low, X_test, Y_test \
                    = ex2D.get_curve2(num_hf=i, num_lf=80)

                model.fit(X_train_hf)
                mses[-1] += model.get_mse(X_test, Y_test) / 5

        plt.plot(np.array(ran), mses, 'v-', label='random data')
        plt.xticks(list(ran))
        plt.legend()
        plt.title(titles[j])
        plt.show()
