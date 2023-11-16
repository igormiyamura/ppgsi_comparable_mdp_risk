
import numpy as np

class VAR:
    def __init__(self) -> None:
        pass
    
    def get_value_function(self, p: float, c: float) -> float:
        pass
            
    def get_equivalent_cost(self, p: float, c: float, p_line: float, lim: str='inf') -> float:
        if lim == 'inf':
            return 
        elif lim == 'sup':
            return self.get_empirical_equivalent_cost(p, c, p_line, alpha=0.999, summation=False, continuous=True, timestep=None)
       
    def get_empirical_equivalent_cost(self, p: float, c: float, p_line: float, alpha: float, summation: bool=False, continuous: bool=False, timestep: float=0.1) -> float:
        if summation:
            if continuous:
                t = self._get_continuous_time_to_var(p, alpha, timestep)
                t_line = self._get_continuous_time_to_var(p_line, alpha, timestep)
            else:
                t = self._get_discrete_time_to_var(p, alpha)
                t_line = self._get_discrete_time_to_var(p_line, alpha)
        else:
            t = np.emath.logn((1-p), (alpha * p)) * p
            t_line = np.emath.logn((1-p_line), (alpha * p_line)) * p_line
            # t = (np.log((1 - alpha) * p) / np.log(1 - p)) * p
            # t_line = (np.log((1 - alpha) * p_line) / np.log(1 - p_line)) * p_line
            
            if not continuous:
                t = np.round(t, 0)
                t_line = np.round(t_line, 0)
        
        return t * c / t_line

    def _get_discrete_time_to_var(self, p: float, alpha: float) -> float:
        t, _alpha = 1, 1

        while _alpha > (1 - alpha):
            _alpha = p * (1 - p)**(t-1)
            t += 1
            
        return t
    
    def _get_continuous_time_to_var(self, p: float, alpha: float, timestep: float=0.1) -> float:
        t, _alpha = 1, 1

        while _alpha > (1 - alpha):
            _alpha = p * (1 - p)**(t-1)
            t += timestep
            
        return t
    