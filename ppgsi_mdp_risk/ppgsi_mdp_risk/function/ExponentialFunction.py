
import numpy as np

class ExponentialFunction:
    def __init__(self) -> None:
        pass
    
    def get_value_function(self, vl_lambda: float, p: float, c: float) -> float:
        return np.sign(vl_lambda) * np.exp(vl_lambda * c) * p / (1 - np.exp(vl_lambda * c) * (1 - p))
            
    def get_equivalent_cost(self, p: float, c: float, p_line: float, lim: str='inf') -> float:
        if lim == 'sup':
            l_extreme = self._get_lambda_extreme(p, c)
            return -np.log(1-p_line) / l_extreme
        elif lim == 'inf':
            return c
        else:
            raise Exception('Limite não definido.')
       
    def get_empirical_equivalent_cost(self, p: float, c: float, p_line: float, l: float) -> float:
        v1 = -np.exp(l*c) * p
        v2 = (np.exp(l*c)*p_line - p_line - np.exp(l*c)*p)
        return np.log(v1/v2) * 1/l

    def _get_lambda_extreme(self, p: float, c: float) -> float:
        return -np.log((1-p))/c
    