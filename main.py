import numpy as np
from sklearn.metrics import mean_squared_error
from src import get_example_data1, get_example_data2, get_example_data3, get_example_data4
from src import DataAugmentationGP


if __name__ == "__main__":
    X_train_hf, X_train_lf, y_train_lf, f_high, f_low, X_test, y_test = get_example_data4()

    # create, train, test model
    model = DataAugmentationGP(
        tau=.001, n=0, input_dim=1, f_high=f_high, f_low=f_low
    )

    model.fit(hf_X=X_train_hf)

    predictions = model.predict_means(X_test)

    mse = mean_squared_error(y_true=y_test, y_pred=predictions)
    print('mean squared error: {}'.format(mse))
    model.plot_forecast(1.5)    