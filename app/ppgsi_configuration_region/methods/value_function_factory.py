
from .risk_neutral import RiskNeutral
from .exponential_utility_function import ExponentialUtilityFunction
from .piecewise_linear_transformation import PiecewiseLinearTransformation
from .value_at_risk import ValueAtRisk
from .conditional_value_at_risk import ConditionalValueAtRisk
from .cumulative_distribution_function import CumulativeDistributionFunction

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
        elif method == "piecewise_linear_transformation":
            return PiecewiseLinearTransformation()
        elif method == "value_at_risk":
            return ValueAtRisk()
        elif method == "conditional_value_at_risk":
            return ConditionalValueAtRisk()
        elif method == "cumulative_distribution_function":
            return CumulativeDistributionFunction()
        else:
            raise ValueError(f"Unknown method: {method}")