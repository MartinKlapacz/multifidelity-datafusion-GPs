from src.MFDataFusion import MultifidelityDataFusion
import threading


class MethodAssessment:
    def __init__(self, models, X_test, y_test):
        assert type(models) is list
        for model in models:
            assert isinstance(model, MultifidelityDataFusion)
        self.models = models
        self.X_test = X_test
        self.y_test = y_test

    def fit_models(self, X_train):
        for model in self.models:
            model.fit(hf_X=X_train)

    def adapt_models(self, a, b, adapt_steps, plot=None):
        for model in self.models:
            model.adapt(a, b, adapt_steps, plot=plot,
                              X_test=self.X_test, 
                              Y_test=self.y_test)

    def log_mse(self):
        mses = []
        for model in self.models:
            mses.append(model.assess_log_mse(self.X_test, self.y_test))
        return mses
