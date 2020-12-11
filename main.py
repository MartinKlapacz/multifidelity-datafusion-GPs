import numpy as np
from src import get_example_data1, get_example_data2, get_example_data3, get_example_data4
from src import DataAugmentationGP


if __name__ == "__main__":
    X_train_hf, X_train_lf, y_train_lf, f_high, f_low, X_test, y_test = get_example_data4()

    # create, train, test model
    model = DataAugmentationGP(
        tau=.001, n=0, input_dim=1, f_high=f_high, lf_X=X_train_lf, lf_Y=y_train_lf, adapt_steps=5, lf_hf_adapt_ratio=1
        # tau=.001, n=0, input_dim=1, f_high=f_high, f_low=f_low
    )

    model.fit(hf_X=X_train_hf)

    model.adapt(5)

    model.plot_forecast(5)