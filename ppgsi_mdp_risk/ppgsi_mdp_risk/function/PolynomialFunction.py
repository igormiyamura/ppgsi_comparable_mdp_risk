
import numpy as np

class PolynomialFunction:
    def __init__(self) -> None:
        pass
    
    def get_value_function(self, p: float, c: float, beta: int=2) -> float:
        """Define the value function for Polynomial Utility Function.

        Args:
            p (float): probability
            c (float): cost
            beta (int, optional): beta defining the exponential number. Defaults to 2.

        Returns:
            float: value of value function
        """
        
        try:
            if beta == 1:
                return 1
            elif beta == 1:
                return (c/p)
            elif beta == 2:
                return (c**beta * (2 - p)) / (p**beta)
            elif beta == 3:
                return (c**beta * (p**2 - 6*p + 6)) / (p**beta)
            else:
                raise Exception(f'[get_value_function]: Beta [{beta}] not found.')
        except Exception as e:
            print(f'[get_value_function]: Error - [{e}]')
            
    def get_equivalent_cost(self, p: float, c: float, p_line: float, beta: int=2) -> float:
        """Calculate the equivalent cost for a probability (p), a cost (c), an equivalent probability (p\') and
        a beta.

        Args:
            p (float): probability
            c (float): cost
            p_line (float): arbitrary probability
            beta (int, optional): exponential number. Defaults to 2.

        Raises:
            Exception: beta not identified.

        Returns:
            float: equivalent cost
        """
        
        if beta == 0:
            return 1
        # elif beta == 0.1:
        #     v = (np.power(c, beta) * p * mpmath.polylog((1-p), -beta) * (1 - p_line)) / ((1 - p) * p_line * mpmath.polylog((1-p_line), -beta))
        #     return np.power(v, 1/beta)
        elif beta == 1:
            return c * p_line / p
        elif beta == 2:
            return np.sqrt((c**2 * p_line**2 * (2 - p)) / (p**2 * (2 - p_line)))
        elif beta == 3:
            eq_p = (p**2 - 6*p + 6)
            eq_p_line = (p_line**2 - 6*p_line + 6)
            v = (c**3 * eq_p * p_line**3) / (p**3 * eq_p_line)
            return np.power(v, 1/beta)
        elif beta == 10:
            v = (c**beta * p_line**beta * self._func_beta_10(p)) / (p**beta * self._func_beta_10(p_line))
            return np.power(v, 1/beta)
        elif beta == 20:
            v = (c**beta * p_line**beta * self._func_beta_20(p)) / (p**beta * self._func_beta_20(p_line))
            return np.power(v, 1/beta)
        else:
            raise Exception(f'[get_value_function]: Beta [{beta}] not found.')
        
    def _func_beta_10(self, p: float) -> float:
        """Auxiliar function to calculate beta = 10 expansion.

        Args:
            p (float): probability

        Returns:
            float: value of expansion
        """        
        return -p**9 + 1022*p**8 - 55980*p**7 + 818520*p**6 - 5103000*p**5 + 16435440*p**4 - \
            29635200*p**3 + 30240000*p**2 - 16329600*p + 3628800

    def _func_beta_20(self, p: float) -> float:
        """Auxiliar function to calculate beta = 20 expansion.

        Args:
            p (float): probability

        Returns:
            float: value of expansion
        """        
        return (-p**19 + 1048574*p**18 - 3483638676*p**17 + 1085570781624*p**16 - 89904730860000*p**15 + \
                    3100376804676480*p**14 - 56163512390086080*p**13 + 611692004959217280*p**12 - 4358654246117808000*p**11 + \
                    21473732319740064000*p**10 - 75875547089306764800*p**9 + 196877625020902425600*p**8 - \
                    380275818414395904000*p**7 + 549443323130397696000*p**6 - 591499300737945600000*p**5 + \
                    467644314338353152000*p**4 - 263665755136143360000*p**3 + 100357207837286400000*p**2 - \
                    23112569077678080000*p + 2432902008176640000)
        
    def get_empirical_equivalent_cost(self, p: float, c: float, p_line: float, beta: int=2, th:float=0.1, _quiet: bool=True) -> float:
        """Get the empirical equivalent cost from Polynomial Utility Function, the return of this function must be equal to the
        [get_equivalent_cost] for the same parameters.

        Args:
            p (float): probability
            c (float): cost
            p_line (float): arbitrary probability
            beta (int, optional): exponential number. Defaults to 2.
            th (float, optional): threashold for probability convergence. Defaults to 0.1.
            _quiet (bool, optional): debug funtionality. Defaults to True.

        Returns:
            float: equivalent cost
        """        
        sum_prob_p = self._polynomial_sum_prob_converge(p, beta, th)
        sum_prob_pline = self._polynomial_sum_prob_converge(p_line, beta, th)
        
        value = (c**beta * p * sum_prob_p) / (p_line * sum_prob_pline)
        
        if not _quiet: 
            print(f"""
            Cost PF - p: {p} | 
            c: {c} | 
            p\': {p_line} | 
            Numerador: {round((c**beta * p * sum_prob_p), 2)} | 
            Denominador: {round((p_line * sum_prob_pline), 2)} | 
            Value: {round(value, 2)} | 
            C\': {round(np.power(value, 1/beta), 2)}""")
        
        return np.round(np.power(value, 1/beta), 4)
        
    def _empirical_value_function(self, c: float, p: float, beta: int, th: float=0.1) -> float:
        """Calculate the empirical value function of Polynomial Utility Function.

        Args:
            c (float): cost
            p (float): probability
            beta (int): exponential number
            th (float, optional): threashold for sum convergence. Defaults to 0.1.

        Returns:
            float: value of value function
        """        
        sum_prob = self._polynomial_sum_prob_converge(p, beta, th)
        return c**beta * p * sum_prob
        
    def _polynomial_sum_prob_converge(self, p: float, beta: int, th: float=0.1) -> float:
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