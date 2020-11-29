import numpy as np
from dataAugmentationGP import DataAugmentationGP
from sklearn.metrics import mean_squared_error
from datasets import get_example_data


if __name__ == "__main__":
    X_train_hf, X_train_lf, y_train_hf, y_train_lf, X_test, y_test = get_example_data()
    def f_low(t): return np.sin(8 * np.pi * t)

    # create, train, test model
    model = DataAugmentationGP(tau=.001, n=4, input_dims=1, f_low=f_low)
    model.fit(hf_X=X_train_hf, hf_Y=y_train_hf)
    predictions = model.predict_means(X_test)
    mse = mean_squared_error(y_true=y_test, y_pred=predictions)
    print('mean squared error: {}'.format(mse))
    model.plot()
