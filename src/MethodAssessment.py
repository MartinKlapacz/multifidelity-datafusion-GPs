from src.MFDataFusion import MultifidelityDataFusion
import threading
import matplotlib.pyplot as plt


class MethodAssessment:
    def __init__(self, models, X_test, Y_test):
        assert type(models) is list
        for model in models:
            assert isinstance(model, MultifidelityDataFusion)
        self.models = models
        self.X_test = X_test
        self.Y_test = Y_test

    def fit_models(self, X_train):
        for model in self.models:
            model.fit(hf_X=X_train)

    def adapt_models(self, adapt_steps, plot_mode=None):
        assert plot_mode in [None, 'e']
        for model in self.models:
            model.adapt(adapt_steps, plot_mode=plot_mode,
                              X_test=self.X_test, 
                              Y_test=self.Y_test)

    def plot(self):
        for model in self.models:
            model.plot()

    def plot_forecast(self, forecast_range):
        for model in self.models:
            model.plot_forecast(forecast_range)

    def mses(self):
        mses = {}
        for model in self.models:
            mses[model.name] = model.get_mse(self.X_test, self.Y_test)
        return mses