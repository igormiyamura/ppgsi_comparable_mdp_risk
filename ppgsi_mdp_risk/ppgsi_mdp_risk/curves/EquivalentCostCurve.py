
import numpy as np

class EquivalentCostCurve:
    def __init__(self, obj_function, rini_p=0.05, rend_p=1, step_p=0.05) -> None:
        self._obj_function = obj_function
        self.range_prob = np.arange(rini_p, rend_p, step_p)
    
    def get_limit_curve(self):
        pass
    
    def get_empirical_limit_curve(self, c: float, p: float, function_type: str, **kwargs) -> dict:
        """Calculate the curve for equivalent cost for a range of probabilities.

        Args:
            c (float): reference cost
            p (float): reference probability
            function_type (str): name of function (method)

        Returns:
            dict: curve for each probability
        """        
        res = {}
        
        for p_line in self.range_prob:
            p_line_rounded = round(p_line, 2)
            
            res[p_line_rounded] = \
                self._obj_function.get_empirical_equivalent_cost(
                    p=p, 
                    c=c, 
                    p_line=p_line, 
                    beta=kwargs['beta'], 
                    th=kwargs['th'], 
                    _quiet=True
                )
                
        return res
            
    