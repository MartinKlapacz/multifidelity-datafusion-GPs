#!./env/bin/python
from src.data.rosenbrock import rosenbrock
import matplotlib.pyplot as plt
import numpy as np

from src import MultifidelityDataFusion
from src import MethodAssessment
from src import NARGP, GPDF, GPDFC

dim = 2
tau = .01


_, _, _, f_exact, f_low, X_test, Y_test = rosenbrock(20, 80, dim)

nargp = NARGP(
    name='NARGP',
    input_dim=dim,
    f_exact=f_exact,
    f_low=f_low,
)

gpdf2 = GPDF(
    name='GPDF2',
    input_dim=dim,
    tau=tau,
    num_derivatives=2,
    f_exact=f_exact,
    f_low=f_low,
)

gpdfc2 = GPDFC(
    name='GPDFC2',
    input_dim=dim,
    tau=tau,
    num_derivatives=2,
    f_exact=f_exact,
    f_low=f_low,
)

gpdf4 = GPDF(
    name='GPDF4',
    input_dim=dim,
    tau=tau,
    num_derivatives=4,
    f_exact=f_exact,
    f_low=f_low,
)

gpdfc4 = GPDFC(
    name='GPDFC4',
    input_dim=dim,
    tau=tau,
    num_derivatives=4,
    f_exact=f_exact,
    f_low=f_low,
)

models = [gpdf2, gpdfc2, gpdf4, gpdfc4]

labels = ['gpdf2', 'gpdfc2', 'gpdf4', 'gpdfc4']
sizes = [5, 10, 15, 20, 25]
nruns = 1

mses = np.zeros((len(models), len(sizes)))

for i, model in enumerate(models):
    for j, size in enumerate(sizes):
        mse_runs = []
        # make average runs
        for _ in range(nruns):
            X_train_hf, _, _, _, _, _, _ = rosenbrock(size, 80, dim)
            model.fit(X_train_hf)
            new_mse = model.get_mse(X_test, Y_test)
            mse_runs.append(new_mse)
        
        mses[i, j] = np.array(mse_runs).mean()
        print(size, model, mse_runs)


for i in range(len(mses)):
    plt.plot(sizes, mses[i,:], '>-', label=labels[i])
    plt.xticks(sizes)

plt.yscale('log')
plt.xlabel('hf-points', fontsize=16)
plt.ylabel('mse', fontsize=16)
plt.title('Rosenbrock function {}D'.format(dim), fontsize=20)
plt.legend()
plt.show()
print(mses)