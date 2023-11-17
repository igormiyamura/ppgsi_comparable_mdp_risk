
import numpy as np

class ConditionalVAR:
    def __init__(self) -> None:
        pass
    
    def get_value_function(self, p: float, c: float) -> float:
        pass
            
    def get_equivalent_cost(self, p: float, c: float, p_line: float, lim: str='inf') -> float:
        if lim == 'inf':
            return (2*c)/(1+c)
        elif lim == 'sup':
            return (1-p)/(1-p_line)
       
    def get_empirical_equivalent_cost(self, p: float, c: float, p_line: float, alpha: float, summation: bool=False, continuous: bool=False, timestep: float=0.1) -> float:
        if summation:
            if continuous:
                t = self._get_continuous_time_to_var(p, alpha, timestep)
                t_line = self._get_continuous_time_to_var(p_line, alpha, timestep)
                
                cvar = self._get_conditional_var_converge(c, p, t, timestep)
                cvar_line = self._get_conditional_var_converge(c, p_line, t_line, timestep)
            else:
                t = self._get_discrete_time_to_var(p, alpha)
                t_line = self._get_discrete_time_to_var(p_line, alpha)
                
                cvar = self._get_conditional_var_converge(c, p, t)
                cvar_line = self._get_conditional_var_converge(c, p_line, t_line)
        else:
            t = self._get_log_nth_term(p, alpha)
            t_line = self._get_log_nth_term(p_line, alpha)
            
            cvar = self._get_sum_infinite_gp(c, p, t)
            cvar_line = self._get_sum_infinite_gp(c, p_line, t_line)
            
            
            if not continuous:
                t = np.round(t, 0)
                t_line = np.round(t_line, 0)
        
        return ((c * t) + cvar) / (t_line + cvar_line)

    def _get_conditional_var_converge(self, c: float, p: float, t: float, timestep: float=1):
        _delta, cvar, error = np.inf, 0, 0.001
        summation = 0
        
        while error > (_delta):
            summation += (1-p)**t
            cvar_ant = cvar
            cvar = c * p * (summation)
            
            _delta = cvar - cvar_ant
            t += timestep
        
        return cvar

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
    
    def _get_log_nth_term(self, p: float, alpha: float):
        return np.emath.logn((1-p), alpha * p - p + 1) 
    
    def _get_sum_infinite_gp(self, c: float, p: float, t: float):
        return c / (1 - p)**(t-1)