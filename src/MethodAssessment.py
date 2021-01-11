from src.MFDataFusion import MultifidelityDataFusion
import threading


class MethodAssessment:
    def __init__(self, regressor1, regressor2, X_test, y_test):
        assert isinstance(regressor1, MultifidelityDataFusion)
        assert isinstance(regressor2, MultifidelityDataFusion)

        self.regressor1 = regressor1
        self.color1 = 'b'

        self.regressor2 = regressor2
        self.color2 = 'r'

        self.X_test = X_test
        self.y_test = y_test

    def fit_models(self, X_train):
        self.regressor1.fit(hf_X=X_train)
        self.regressor2.fit(hf_X=X_train)

    def adapt_models(self, a, b, adapt_steps, plot=None):
        self.regressor1.adapt(a, b, adapt_steps, plot=plot,
                              X_test=self.X_test, Y_test=self.y_test)
        self.regressor2.adapt(a, b, adapt_steps, plot=plot,
                              X_test=self.X_test, Y_test=self.y_test)

    def log_mse(self):
        mse1 = self.regressor1.assess_log_mse(self.X_test, self.y_test)
        mse2 = self.regressor2.assess_log_mse(self.X_test, self.y_test)
        return mse1, mse2
