import threading
import numpy as np
import concurrent.futures
import logging
import time
from scipy.optimize import fmin

def timer(func):
    def wrapper(*args):
        start = time.time()
        res = func(*args)
        end = time.time()
        print("hf point acquisition duration: {}".format(end - start))
        return res
    return wrapper

def f(x):
    return x**2

start_positions = [1,2,3,4,5]

def foo():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda sp: fmin(f, sp, full_output=True), start_positions))

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    foo()