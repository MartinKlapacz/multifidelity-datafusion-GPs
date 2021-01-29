import abc

class AbstractMFGP(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def fit(self, hf_X):
        pass

    @abc.abstractmethod
    def adapt(self, adapt_steps, plot_mode, X_test, Y_test):
        pass

    @abc.abstractmethod
    def predict(self, X_test):
        pass

    @abc.abstractmethod
    def plot(self):
        pass

    @abc.abstractmethod
    def plot_forecast(self, forecast_range):
        pass

    @abc.abstractmethod
    def get_mse(self, X_test, Y_test):
        pass