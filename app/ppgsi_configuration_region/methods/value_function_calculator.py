

from abc import ABC, abstractmethod

class ValueFunctionCalculator(ABC):
    """Abstract base class for calculating the value function using different methods."""

    @abstractmethod
    def osma_analytical_value_function(self, *args, **kwargs):
        """Calculate the value function based on the given method.

        Args:
            *args: Positional arguments for the calculation method.
            **kwargs: Keyword arguments for the calculation method.

        Returns:
            The result of the value function calculation.
        """
        pass

    @abstractmethod
    def mss_value_function(self, *args, **kwargs):
        """Calculate the value function based on the given method.

        Args:
            *args: Positional arguments for the calculation method.
            **kwargs: Keyword arguments for the calculation method.

        Returns:
            The result of the value function calculation.
        """
        pass
    
    @abstractmethod
    def mss_analytical_value_function(self, *args, **kwargs):
        """Calculate the analytical value function based on the given method.

        Args:
            *args: Positional arguments for the calculation method.
            **kwargs: Keyword arguments for the calculation method.

        Returns:
            The result of the value function calculation.
        """
        pass
    
    @abstractmethod
    def osma_value_function_range_probability(self, *args, **kwargs):
        """Calculate the analytical value function based on the given method.

        Args:
            *args: Positional arguments for the calculation method.
            **kwargs: Keyword arguments for the calculation method.

        Returns:
            The result of the value function calculation.
        """
        pass
    
    @abstractmethod
    def mss_value_function_range_probability(self, *args, **kwargs):
        """Calculate the analytical value function based on the given method.

        Args:
            *args: Positional arguments for the calculation method.
            **kwargs: Keyword arguments for the calculation method.

        Returns:
            The result of the value function calculation.
        """
        pass