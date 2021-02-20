import abc


class AbstractGPC(metaclass=abc.ABCMeta):

    def __init__(self, function: callable):
        self.function = function

    @abc.abstractmethod
    def update_order(self, new_order):
        pass

    @abc.abstractmethod
    def calculate_coefficients(self):
        pass

    @abc.abstractmethod 
    def get_mean(self):
        pass 

    @abc.abstractmethod
    def get_var(self):
        pass

    def get_mean_var(self):
        return self.get_mean(), self.get_var()

    def update_function(self, function):
        self.function = function
        self.calculate_coefficients()