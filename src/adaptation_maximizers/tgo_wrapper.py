import numpy as np
from .abstract_maximizer import AbstractMaximizer
from tgo import tgo


class TGOMaximizer(AbstractMaximizer):
    """Wrapper class for the uncertainty maximization in the adaptation 
    process, uses the deterministic SLSQP global optimization algorithm
    to solve the optimization problem. It wraps TGO 
    https://stefan-endres.github.io/tgo/
    """

    def __init__(self):
        super().__init__()

    def maximize(self, model_predict: callable, lower_bound: np.ndarray, upper_bound: np.ndarray):
        bound = []
        dim = len(lower_bound)
        for i in range(dim):
            bound.append((lower_bound[i], upper_bound[i]))

        def acquisition_curve(x: float):
            _, uncertainty = model_predict(x[None])
            return - uncertainty[:, None]

        res = tgo(acquisition_curve, bound)
        print("Selected point", res.x, res.fun)
        # x = np.atleast_2d(np.linspace(lower_bound, upper_bound, 100))
        # _, u = model_predict(x)
        # print("Other vals = ", np.max(u))
        return res.x, res.fun
