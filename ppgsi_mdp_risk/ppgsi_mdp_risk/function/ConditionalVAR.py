
import numpy as np

class ConditionalVAR:
    def __init__(self) -> None:
        pass
    
    def get_value_function(self, p: float, c: float) -> float:
        pass
            
    def get_equivalent_cost(self, p: float, c: float, p_line: float, alpha: float, continuous: bool=False) -> float:
        pass
       
    def get_empirical_equivalent_cost(self, p: float, c: float, p_line: float, alpha: float, continuous: bool=False, timestep: float=0.1) -> float:
        if continuous:
            t = self._get_continuous_time_to_var(p, alpha, timestep)
            t_line = self._get_continuous_time_to_var(p_line, alpha, timestep)
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
    
    def _get_continuous_time_to_var(self, p: float, alpha: float, timestep: float=0.1) -> float:
        t, _alpha = 1, 1

        while _alpha > (1 - alpha):
            _alpha = p * (1 - p)**(t-1)
            t += timestep
            
        return t
    
    def _cvar_sum_prob_converge(self, p: float, beta: int, th: float=0.1) -> float:
        """Calculate the convergence of sum of probabilities in Polynomial Utility Function equation.

        Args:
            p (float): probability
            beta (int): exponential number
            th (float, optional): threashold. Defaults to 0.1.

        Returns:
            float: value of convergence
        """        
        error, sum_prob = np.inf, 0
        t = 1
        while (error > th) or (t < 1000):
            sum_prob_ant = sum_prob
            sum_prob += t**beta * (1 - p)**(t-1)
            
            error = sum_prob - sum_prob_ant
            t += 1
            
        return sum_prob