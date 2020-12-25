import numpy as np
from src import get_example_data1, get_example_data2, get_example_data3, get_example_data4
from src import DataAugmentationGP
import GPy
import matplotlib.pyplot as plt


if __name__ == "__main__":
    X_train_hf, X_train_lf, y_train_lf, f_high, f_low, X_test, y_test = get_example_data4()

    # create, train, test model
    model = DataAugmentationGP(
        # data driven
        # input_dim=1, tau=.001, n=1, f_high=f_high, adapt_steps=5, lf_X=X_train_lf, lf_Y=y_train_lf, lf_hf_adapt_ratio=1
        # function driven
        input_dim=1, tau=.001, n=0, f_high=f_high, adapt_steps=15, f_low=f_low
    )

    model.fit(hf_X=X_train_hf)

    print(model.assess_log_mse(X_test, y_test))

    model.adapt(plot='uncertainty', X_test=X_test, Y_test=y_test)

    print(model.assess_log_mse(X_test, y_test))

    # model.plot_forecast(5)
    plt.show()