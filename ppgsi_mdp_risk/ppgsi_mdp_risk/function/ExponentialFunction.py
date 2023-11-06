
import numpy as np

class ExponentialFunction:
    def __init__(self) -> None:
        pass
    
    def get_value_function(self, p: float, c: float, beta: int=2) -> float:
        return 0
            
    def get_equivalent_cost(self, p: float, c: float, p_line: float, beta: int=2) -> float:
        return 0
       
    def get_empirical_equivalent_cost(self, p: float, c: float, p_line: float, l: float) -> float:
        v1 = -np.exp(l*c)*p
        v2 = (np.exp(l*c)*p_line - p_line - np.exp(l*c)*p)
        return np.log(v1/v2) * 1/l

    def _get_lambda_extreme(self, p: float, c: float) -> float:
        return -np.log((1-p))/c
    