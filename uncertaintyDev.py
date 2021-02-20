#!./env/bin/python
import numpy as np
import GPy
import matplotlib.pyplot as plt
import time

from src import example1D
from src import example2D
from src import MultifidelityDataFusion
from src import MethodAssessment
from src import NARGP, GPDF, GPDFC


if __name__ == "__main__":
    X_train_hf, X_train_lf, Y_train_lf, f_exact, f_low, X_test, Y_test \
        = example1D.get_curve4(num_hf=5, num_lf=100)

    # create, train, test model
    model = NARGP(
        input_dim=1,
        f_exact=f_exact,
        f_low=f_low,
    )

    model.fit(hf_X=X_train_hf)
    model.adapt(15, 'u', X_test, Y_test)
    plt.show()