def augmentIter(n):
    # generates a number sequence 0, -1, 1, -2, 2, ..., -n, n
    if n == 0:
        yield 0
    else:
        i = 0
        sign = -1
        while i < n or sign == 1:
            if i == 0:
                yield 0
            if sign == 1:
                sign = -1
            else:
                sign = 1
                i += 1
            yield sign * i