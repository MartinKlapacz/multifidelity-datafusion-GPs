import numpy as np
from .abstract_maximizer import AbstractMaximizer
from DIRECT import solve


class DIRECT1Maximizer(AbstractMaximizer):
    """Wrapper class for the uncertainty maximization in the adaptation 
    process, uses the deterministic DIRECT-1 global optimization algorithm
    to solve the optimization problem
    """

    def __init__(self):
        super().__init__()
        # DIRECT.solve params
        self.maxT = 50
        self.algmethod = 1

    def maximize(self, model_predict: callable, lower_bound: np.ndarray, upper_bound: np.ndarray):

        def acquisition_curve(x: float, dummy):
            # DIRECT.solve() calls this function with x and a dummy value
            _, uncertainty = model_predict(x[None])
            return - uncertainty[:, None]

        xopt, fopt, _ = solve(acquisition_curve, lower_bound, upper_bound,
                              maxT=self.maxT, algmethod=self.algmethod)
        return xopt, fopt
