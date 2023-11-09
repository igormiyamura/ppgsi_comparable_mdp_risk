
import numpy as np

class PiecewiseTransformation:
    def __init__(self) -> None:
        pass
    
    def get_value_function(self, p: float, c: float, beta: int=2) -> float:
        return 0
            
    def get_equivalent_cost(self, p: float, c: float, p_line: float, lim: str='inf') -> float:
        if lim == 'inf':
            return (-p*p_line*c + p_line*c) / (-p*p_line + p)
        elif lim == 'sup':
            return c
        else:
            raise Exception('Limite não definido.')
       
    def get_empirical_equivalent_cost(self, p: float, c: float, p_line: float, k: float, gamma: float, alpha: float) -> float:
        v1 = c * (2*k*p - k + 1) 
        v2 = (gamma - 1) * (k + p - 1) + p - gamma * k * p
        v3 = (gamma - 1) * (k + p_line - 1) + p_line - gamma * k * p_line
        v4 = 2 * k * p_line - k + 1
        return v1 / v2 * v3 / v4

    def _get_alpha(self, k: float) -> float:
        return np.round(1/(1+abs(k)), 2)