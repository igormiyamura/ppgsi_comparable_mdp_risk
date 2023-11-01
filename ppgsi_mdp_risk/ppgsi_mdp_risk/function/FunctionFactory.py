
from ppgsi_mdp_risk.function import PolynomialFunction

class FunctionFactory:
    def __init__(self) -> None:
        self.polynomial_function = PolynomialFunction.PolynomialFunction()
    
    def get_function(self, nm_function: str):
        """Get the object of function used to evaluate the MDP.

        Args:
            nm_function (str): name of the function

        Raises:
            Exception: function name not identified

        Returns:
            object: object of function
        """        
        if nm_function == 'PolynomialFunction':
            return self.polynomial_function
        else:
            raise Exception(f'[get_function]: Function name not identified [{nm_function}]')