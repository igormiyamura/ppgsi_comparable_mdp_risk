
import numpy as np

class ConditionalVAR:
    def __init__(self) -> None:
        pass
    
    def get_value_function(self, p: float, c: float) -> float:
        pass
            
    def get_equivalent_cost(self, p: float, c: float, p_line: float, lim: str='inf') -> float:
        if lim == 'inf':
            #return (2*c)/(1+c)
            return (c * p_line * (1 - p)) / (p * (1 - p_line))
        elif lim == 'sup':
            #return (1-p)/(1-p_line)
            return c * p_line / p
       
    def _get_t_term(self, c, p, alpha, summation: bool=False, continuous: bool=False, timestep: float=0.1) -> float:
        if summation:
            if continuous:
                t = self._get_continuous_time_to_var(p, alpha, timestep)
                cvar = self._get_conditional_var_converge(c, p, t, timestep)
            else:
                t = self._get_discrete_time_to_var(p, alpha)
                cvar = self._get_conditional_var_converge(c, p, t)
        else:
            t = self._get_log_nth_term(p, alpha)
            cvar = self._get_sum_infinite_gp(c, p, t)
            
        return t, cvar
       
    def get_empirical_equivalent_cost(self, p: float, c: float, p_line: float, alpha: float, summation: bool=False, continuous: bool=False, timestep: float=0.1) -> float:
        t, cvar = self._get_t_term(c, p, alpha, summation, continuous, timestep)
        t_line, cvar_line = self._get_t_term(c, p_line, alpha, summation, continuous, timestep)
        
        # return ((c * t) + cvar) / (t_line + cvar_line)
        return c * (1 - t * (1 - alpha) * p) / (p * alpha) * (p_line * alpha) / (1 - (t_line * (1-alpha) * p_line))
        # return c * np.sqrt(p_line * t / p * t_line)
        # return (c * t)/t_line
        # return ((1-t*p*p) / (p*(1-p))) * ((p_line*(1-p_line)) / 1-(t_line*p_line*p_line))

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
        # return np.emath.logn((1-p), alpha * p - p + 1) 
        return np.emath.logn((1-p), alpha * (1 - p) / p)
        # return np.emath.logn(1-alpha, 1-p)
    
    def _get_sum_infinite_gp(self, c: float, p: float, t: float):
        return c / (1 - p)**(t-1)