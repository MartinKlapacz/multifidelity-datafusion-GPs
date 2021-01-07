import abc
import numpy as np


class AbstractAugmIterator(metaclass=abc.ABCMeta):
    def __iter__(self):
        return self

    @abc.abstractmethod
    def __next__(self):
        pass

    @abc.abstractmethod
    def new_entries_count(self):
        pass

    @abc.abstractmethod
    def reset(self):
        pass


class EvenAugmentation(AbstractAugmIterator):
    # generates a number sequence 0, 1, -1, 2, -2, ..., n, -n
    def __init__(self, n, dim=1):
        self.reset()
        self.n = n
        self.dim = dim
        self.dim_i = 0
        self.sign = -1

    def __next__(self):
        vector = np.zeros(self.dim)
        if self.i == 0:
            self.i = 1
            return vector
        if self.i <= self.n:
            if self.sign == -1:
                vector[self.dim_i] = -self.i
                if self.dim_i == self.dim - 1:
                    self.dim_i = 0
                    self.sign = 1
                else:
                    self.dim_i += 1
                return vector 
            if self.sign == 1:
                vector[self.dim_i] = self.i
                if self.dim_i == self.dim - 1:
                    self.dim_i = 0
                    self.sign = -1
                    self.i += 1
                else:
                    self.dim_i += 1
                return vector 

        self.reset()
        raise StopIteration

    def new_entries_count(self):
        return 2 * self.n * self.dim + 1

    def reset(self):
        self.i = 0
        self.sign = -1
        self.dim_i = 0


class BackwardAugmentation(AbstractAugmIterator):
    # generates a number sequence 0, -1, -2, ..., -n
    def __init__(self, n, dim=1):
        self.reset()
        self.n = n
        self.dim = dim
        self.dim_i = 0

    def __next__(self):
        vector = np.zeros(self.dim)
        if self.i == 0:
            self.i = 1
            return vector
        if self.i <= self.n:
            vector[self.dim_i] = -self.i
            if self.dim_i == self.dim - 1:
                self.i += 1
                self.dim_i = 0
            else:
                self.dim_i += 1
            return vector
        self.reset()
        raise StopIteration

    def new_entries_count(self):
        return self.n * self.dim + 1

    def reset(self):
        self.i = 0
        self.dim_i = 0