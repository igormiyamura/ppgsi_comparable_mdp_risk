
from .risk_neutral import RiskNeutral
from .exponential_utility_function import ExponentialUtilityFunction

class ValueFunctionFactory:
    """Factory class to create and run value function calculators."""

    @staticmethod
    def get_calculator(method: str):
        """Retrieve a value function calculator based on the specified method.

        Args:
            method (str): The method name (e.g., 'monte_carlo', 'td', 'dp').

        Returns:
            ValueFunctionCalculator: An instance of the specified calculator.
        """
        if method == "risk_neutral":
            return RiskNeutral()
        elif method == "exponential_utility_function":
            return ExponentialUtilityFunction()
        else:
            raise ValueError(f"Unknown method: {method}")