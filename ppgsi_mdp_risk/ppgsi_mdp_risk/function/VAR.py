
import numpy as np

class VAR:
    def __init__(self) -> None:
        pass
    
    def get_value_function(self, p: float, c: float) -> float:
        pass
            
    def get_equivalent_cost(self, p: float, c: float, p_line: float, lim: str='inf') -> float:
        pass
       
    def get_empirical_equivalent_cost(self, p: float, c: float, p_line: float, alpha: float, continuous: bool=False) -> float:
        if continuous:
            t = self._get_discrete_time_to_var(p, alpha)
            t_line = self._get_discrete_time_to_var(p_line, alpha)
        else:
            t = self._get_discrete_time_to_var(p, alpha)
            t_line = self._get_discrete_time_to_var(p_line, alpha)
        
        return t * c / t_line

    def _get_discrete_time_to_var(self, p: float, alpha: float) -> float:
        t, _alpha = 1, 1

        while _alpha > (1 - alpha):
            _alpha = p * (1 - p)**(t-1)
            t += 1
            
        return t
    