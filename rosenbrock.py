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

# Make data.

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

# models = [gpdf2, gpdfc2, gpdf4, gpdfc4]
models = [gpdfc2 ]#, gpdfc2, gpdf4, gpdfc4]

labels = ['gpdf2', 'gpdfc2', 'gpdf4', 'gpdfc4']
sizes = [5, 10, 15, 20, 25]
nruns = 1

mses = np.zeros((len(models), len(sizes)))

for i, model in enumerate(models):
    for j, size in enumerate(sizes):
        mse_runs = []
        for _ in range(nruns):

            X_train_hf, _, Y_train_lf, _, _, _, _ = rosenbrock(size, 80, dim)

            # fig = plt.figure()
            # ax = fig.gca(projection='3d')
            # X1 = np.arange(-0, 1, .05)
            # X2 = np.arange(-0, 1, .05)
            # X1, X2 = np.meshgrid(X1, X2)

            # X = np.vstack((X1.flatten(), X2.flatten())).T
            # Y = f_exact(X)
            # ax.scatter(X1, X2, Y)
            # plt.show()

            print(f_exact(X_train_hf))

            model.fit(X_train_hf)
            model.plot()
            preds = model.predict(X_test)
            plt.scatter(X_train_hf[:,0], X_train_hf[:,1], f_exact(X_train_hf), c='black')
            plt.scatter(X_test[:,0], X_test[:,1], f_exact(X_test), c='purple')
            plt.show()
            new_mse = model.get_mse(X_test, Y_test)
            mse_runs.append(new_mse)
        
        mses[i, j] = np.array(mse_runs).mean()
        print(size, model, mse_runs)

print(mses)

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