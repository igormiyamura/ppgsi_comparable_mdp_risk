
import numpy as np
from typing import List

from ..residuals.error_metrics import ErrorMetrics
from ..mss.multiple_sequential_states import MultipleSequentialStates
from .value_function_calculator import ValueFunctionCalculator

class PiecewiseLinearTransformation(ValueFunctionCalculator):
    def __init__(self) -> None:
        pass

    def osma_analytical_value_function(self, p: float, c: float, k: float):
        return (c * (k - 2 * k * p + 1)) / (p * (1 - k))

    def mss_value_function(self, n: int, p: float, c: float, k: float, alpha: float, _threshold: int, _epsilon: float, _verbose: bool = False):
        """
        Computes the value function for Multiple Sequential States (MSS) with specified parameters.

        Parameters:
        n (int): Number of states.
        p (float): Probability.
        c (float): Cost.
        k (float): Adjustment factor for the piecewise function.
        _threshold (int): Maximum number of iterations.
        _epsilon (float): Convergence threshold.
        _verbose (bool): Flag to suppress debug outputs.

        Returns:
        tuple: Final value function and number of iterations.
        """
        # Verify if alpha is within the valid range
        assert 0 < alpha <= 1/(1 + abs(k)), "Alpha must be within the range (0, 1/(1 + |k|)]"
        
        EM = ErrorMetrics(_epsilon)
        iteration = 0

        # Initialize Multiple Sequential States (MSS)
        MSS = MultipleSequentialStates(num_states=n, probability=p, cost=c)

        # Build initial value function and transition probabilities
        initial_value_function = MSS._build_V0()
        transition_probabilities = MSS.build_transition_probabilities()

        if _verbose:
            display(initial_value_function)
            display(transition_probabilities)

        # Copy initial values to working variables
        current_values = initial_value_function.copy()
        updated_values = initial_value_function.copy()
        transitions = transition_probabilities.copy()

        def piecewise_function(x, k):
            """Applies a piecewise linear adjustment based on the sign of x."""
            if _verbose:
                print('Piecewise Function:', 'x - Negative' if x < 0 else 'x - Positive')
            return (1 - k) * x if x < 0 else (1 + k) * x

        while iteration < _threshold:
            # Save a copy of the previous value function for convergence check
            previous_values = current_values.copy()

            # Iterate through states and actions
            for state in transitions.keys():
                for action in transitions[state].keys():
                    accumulated_q = 0

                    # Compute Q-value for the current state-action pair
                    for next_state in transitions.keys():
                        if _verbose:
                            print(f'[{iteration}] State: {state}, Action: {action}, Next State: {next_state}')

                        # Determine the cost and transition probability
                        cost = 1 if state != 'sG' else 0
                        transition_probability = transitions[state][action][next_state]

                        # Compute the adjustment term for the value function
                        adjustment = (
                            piecewise_function(cost + updated_values[next_state] - updated_values[state], k)
                            if state != 'sG' else 0
                        )

                        # Accumulate the Q-value
                        accumulated_q += transition_probability * adjustment

                        if _verbose:
                            print(f'Accumulated Q: {accumulated_q}')

                    # Update Q-value for the current state
                    updated_values[state] += alpha * (accumulated_q)
                    if _verbose:
                        print(f'Alpha: {alpha} | Accumulated Q: {accumulated_q} | Updated Value: {updated_values[state]}')

                # Update the value function
                current_values[state] = updated_values[state]

            if _verbose:
                print(f'Current Values: {current_values} | Previous Values: {previous_values}')
                print('---')

            iteration += 1

            # Check for convergence
            if EM.relative_residual(
                np.array(list(current_values.values())),
                np.array(list(previous_values.values()))
            ):
                break

        return current_values, iteration

    def mss_analytical_value_function(self, *args, **kwargs):
        pass
    
    def osma_value_function_range_probability(self, p: List[float], c: float, k: float, _verbose: bool = False):
        print(f"""
              Calculando valores para os seguintes parâmetros:
                p: {[round(v, 2) for v in p]} | 
                c: {c} | 
                k: {k} |
              """)
        
        res = {}
        res[1] = {}        
        
        for prob in p:
            prob = round(prob, 2)
            res[1][prob] = self.osma_analytical_value_function(prob, c, k)
            res[1][prob] = np.nan if res[1][prob] > 1e3 else res[1][prob]
                
        return res
    
    def mss_value_function_range_probability(self, n: List[int], p: List[float], c: float, k: float, alpha: float, _threshold: int, _epsilon: float, _verbose: bool = False, _validate_larger_values: bool = False):
        print(f"""
              Calculando valores para os seguintes parâmetros:
                n: {[v for v in n]} | 
                p: {[round(v, 2) for v in p]} | 
                c: {c} | 
                k: {k} |
                alpha: {alpha} |
                threshold: {_threshold} | 
                epsilon: {_epsilon} | 
              """)
        
        res = {}
        
        for num_states in n:
            res[num_states] = {}
            for prob in p:
                prob = round(prob, 2)
                res[num_states][prob], i = self.mss_value_function(num_states, prob, c, k, alpha, _threshold, _epsilon, _verbose)
                if _validate_larger_values: 
                    res[num_states][prob] = np.nan if res[num_states][prob][0] > 1e3 else res[num_states][prob][0]
                else:
                    res[num_states][prob] = res[num_states][prob][0]
                
        return res