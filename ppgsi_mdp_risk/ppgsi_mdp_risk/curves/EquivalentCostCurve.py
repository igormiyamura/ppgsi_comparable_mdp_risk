
import numpy as np

class EquivalentCostCurve:
    def __init__(self, obj_function, nm_function, rini_p=0.05, rend_p=1, step_p=0.05) -> None:
        self._obj_function = obj_function
        self._nm_function = nm_function
        self.range_prob = np.arange(rini_p, rend_p, step_p)
    
    def get_limit_curve(self, c: float, p: float, **kwargs):
        """Calculate the curve for equivalent cost for a range of probabilities.

        Args:
            c (float): reference cost
            p (float): reference probability

        Returns:
            dict: curve for each probability
        """        
        res = {}
        
        for p_line in self.range_prob:
            p_line_rounded = round(p_line, 2)
            
            res[p_line_rounded] = \
                self._obj_function.get_equivalent_cost(
                    p=p, 
                    c=c, 
                    p_line=p_line, 
                    lim=kwargs['lim']
                )
                
        return res
    
    def get_empirical_limit_curve(self, c: float, p: float, **kwargs) -> dict:
        """Calculate the curve for equivalent cost for a range of probabilities.

        Args:
            c (float): reference cost
            p (float): reference probability

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
                    **kwargs
                )
            
        return res
          
    def get_multi_limit_curve(self, c: float, list_p: float, **kwargs) -> dict:
        """Calculate the curve for equivalent cost for a range of probabilities.

        Args:
            c (float): reference cost
            list_p (float): list of probabilities

        Returns:
            dict: curve for each probability
        """        
        res = {}
        
        for p in list_p:
            res[p] = {}
            for p_line in self.range_prob:
                p_line_rounded = round(p_line, 2)
                
                res[p][p_line_rounded] = \
                    self._obj_function.get_equivalent_cost(
                        p=p, 
                        c=c, 
                        p_line=p_line, 
                        lim=kwargs['lim']
                    )
                
        return res
            
    def get_multi_empirical_limit_curve(self, c: float, list_p: float, **kwargs) -> dict:
        """Calculate the curve for equivalent cost for a range of probabilities.

        Args:
            c (float): reference cost
            list_p (float): list of probabilities

        Returns:
            dict: curve for each probability
        """        
        res = {}
        
        for p in list_p:
            res[p] = {}
            for p_line in self.range_prob:
                p_line_rounded = round(p_line, 2)
                
                res[p][p_line_rounded] = \
                    self._obj_function.get_empirical_equivalent_cost(
                        p=p,
                        c=c,
                        p_line=p_line,
                        **kwargs
                    )
                
        return res
    