class EvenAugmentation():
    # generates a number sequence 0, 1, -1, 2, -2, ..., n, -n
    def __init__(self, n):
        self.reset()
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == 0:
            self.i += 1
            return 0
        if self.i <= self.n and self.sign == 1:
            self.sign = -1
            return self.i
        if self.i <= self.n and self.sign == -1:
            val = self.i
            self.sign = 1
            self.i += 1
            return -val
        self.reset() # make same iterator object reuseable
        raise StopIteration

    def numOfNewAugmentationEntries(self):
        return 2 * self.n + 1

    def reset(self):
        self.i = 0
        self.sign = 1