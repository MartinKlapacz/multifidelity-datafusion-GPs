import numpy as np
from .abstract_augm_iterator import AbstractAugmIterator

class EvenAugmentation(AbstractAugmIterator):
    """generates a number sequence 0, 1, -1, 2, -2, ..., n, -n.
    Can be used to build a manifold function with input pattern 
    g(t, f_l(t - n*tau), ..., f_l(t), ..., f_l(t + n*tau)) if dim = 1.
    """

    def __init__(self, n, dim=1):
        super().__init__(n, dim=dim)


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
