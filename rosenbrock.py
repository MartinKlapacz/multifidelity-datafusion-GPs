#!./env/bin/python
from src.data.rosenbrock import rosenbrock
import matplotlib.pyplot as plt
import numpy as np

from src import MultifidelityDataFusion
from src import MethodAssessment
from src import NARGP, GPDF, GPDFC


X_train_hf, X_train_lf, Y_train_lf, f_exact, f_low, X_test, Y_test = rosenbrock(20, 80, 8)


tau = .01

gpdf2 = GPDF(
    name='GPDF2',
    input_dim=8,
    tau=tau,
    num_derivatives=2,
    f_exact=f_exact,
    f_low=f_low,
)

gpdfc2 = GPDFC(
    name='GPDFC2',
    input_dim=8,
    tau=tau,
    num_derivatives=2,
    f_exact=f_exact,
    f_low=f_low,
)

gpdf4 = GPDF(
    name='GPDF4',
    input_dim=8,
    tau=tau,
    num_derivatives=4,
    f_exact=f_exact,
    f_low=f_low,
)

gpdfc4 = GPDFC(
    name='GPDFC4',
    input_dim=8,
    tau=tau,
    num_derivatives=4,
    f_exact=f_exact,
    f_low=f_low,
)

models = [gpdf2, gpdfc2, gpdf4, gpdfc4]
names = ['gpdf2', 'gpdfc2', 'gpdf4', 'gpdfc4']

sizes = [5, 10, 15, 20, 25]

mses = np.zeros((len(models), len(sizes)))

nruns = 5


for i, model in enumerate(models):
    for j, size in enumerate(sizes):
        for _ in range(nruns):

            X_train_hf, X_train_lf, y_train_lf, f_high, f_low, X_test, y_test = rosenbrock(20, 80, 8)

            model.fit(X_train_hf)
            mses[i, j] += model.get_mse(X_test, Y_test) / nruns

for i, row in enumerate(mses):
    plt.plot(sizes, row, '>-', label=names[i])
    plt.xticks(sizes)

plt.yscale('log')
plt.xlabel('hf-points', fontsize=16)
plt.ylabel('mse', fontsize=16)
plt.title('Rosenbrock function 8D', fontsize=20)
plt.legend()
plt.show()
