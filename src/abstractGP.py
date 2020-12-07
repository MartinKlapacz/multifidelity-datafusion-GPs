import abc

class AbstractGP(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def predict(self, X_test):
        pass

    @abc.abstractmethod
    def plot(self):
        pass