import abc

class AbstractGP(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def predict(self, X_test):
        pass

    @abc.abstractmethod
    def plot(self):
        pass

    @abc.abstractmethod
    def plot_forecast(self, forecast_range):
        pass