import numpy as np
from .abstract_augm_iterator import AbstractAugmIterator

class BackwardAugmentation(AbstractAugmIterator):
    """ generates a number sequence 0, -1, -2, ..., -n.
    Can be used to build a manifold function g with input pattern 
    g(t, f_l(t - n*tau), ..., f_l(t), ..., f_l(t + n*tau)) if dim = 1.

    :param n: number of derivatives to include
    :type n: integer
    :param dim: input dimension of the model used to (dim of first param in g), 
    defaults to 1
    :type dim: int, optional
    """

    def __init__(self, n, dim=1):
        super().__init__(n, dim=dim)        

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