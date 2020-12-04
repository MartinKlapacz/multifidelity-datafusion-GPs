import numpy as np
from dataAugmentationGP import DataAugmentationGP
from sklearn.metrics import mean_squared_error
from datasets import get_example_data


if __name__ == "__main__":
    X_train_hf, X_train_lf, y_train_lf, f_high, f_low, X_test, y_test = get_example_data()

    # create, train, test model
    model = DataAugmentationGP(
        tau=.001, n=1, input_dims=1, f_high=f_high, f_low=f_low
    )

    model.fit(hf_X=X_train_hf)

    predictions = model.predict_means(X_test)

    mse = mean_squared_error(y_true=y_test, y_pred=predictions)
    print('mean squared error: {}'.format(mse))
    model.plot_forecast(1)