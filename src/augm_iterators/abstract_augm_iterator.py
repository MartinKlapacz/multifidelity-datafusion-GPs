from abc import ABCMeta, abstractmethod


class AbstractAugmIterator(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, n, dim=1):
        self.reset()
        self.n = n
        self.dim = dim
        self.dim_i = 0
        self.sign = -1

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def new_entries_count(self):
        """returns the number of new entries this iterator would 
        generate in the augmention process

        :return: number of entries
        :rtype: int
        """
        pass

    @abstractmethod
    def reset(self):
        """(re)initializes the the state of the iterator,
        necessary to make the same interator object reusable
        """
        pass
