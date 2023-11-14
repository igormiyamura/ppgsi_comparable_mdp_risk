
from ppgsi_mdp_risk.function import PolynomialFunction, ExponentialFunction, PiecewiseTransformation, VAR, ConditionalVAR

class FunctionFactory:
    def __init__(self) -> None:
        pass
        
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
            return PolynomialFunction.PolynomialFunction()
        elif nm_function == 'ExponentialFunction':
            return ExponentialFunction.ExponentialFunction()
        elif nm_function == 'PiecewiseTransformation':
            return PiecewiseTransformation.PiecewiseTransformation()
        elif nm_function == 'VAR':
            return VAR.VAR()
        elif nm_function == 'ConditionalVAR':
            return ConditionalVAR.ConditionalVAR()
        else:
            raise Exception(f'[get_function]: Function name not identified [{nm_function}]')
    