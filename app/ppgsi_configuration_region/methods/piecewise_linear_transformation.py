
import numpy as np
from typing import List
from tqdm import tqdm

from ..residuals.error_metrics import ErrorMetrics
from ..mss.multiple_sequential_states import MultipleSequentialStates
from .value_function_calculator import ValueFunctionCalculator

from scipy.optimize import fsolve

class PiecewiseLinearTransformation(ValueFunctionCalculator):
    def __init__(self) -> None:
        self.calculated_values = {}

    def osma_analytical_value_function(self, p: float, c: float, k: float):
        return {0: (c * (k - 2 * k * p + 1)) / (p * (1 - k))}

    def mss_generic_value_function(self, n: int, p: float, c: float, k: float, alpha: float, gamma: float, _threshold: int, _epsilon: float, _verbose: bool = False):
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
        if alpha is not None:
            assert 0 < alpha <= 1/(1 + abs(k)), "Alpha must be within the range (0, 1/(1 + |k|)]"
        else:
            alpha = 1
        
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
                        cost = c if state != 'sG' else 0
                        transition_probability = transitions[state][action][next_state]

                        # Compute the adjustment term for the value function
                        adjustment = (
                            piecewise_function(cost + gamma * updated_values[next_state] - updated_values[state], k)
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

        return current_values

    def mss_value_function(self, n: int, p: float, c: float, k: float, alpha: float, gamma: float = 1, _threshold: int = 10000, _epsilon: float = None, _verbose: bool = False):
        # Verify if alpha is within the valid range
        # print(f'Processando para: n: {n}, p: {p}, c: {c}, k: {k}, alpha: {alpha}, gamma: {gamma}, _threshold: {_threshold}, _epsilon: {_epsilon}')
        
        if alpha is not None:
            assert 0 < alpha <= 1/(1 + abs(k)), "Alpha must be within the range (0, 1/(1 + |k|)]"
        else:
            alpha = 1
            
        # Define Transition Matrix
        T = np.hstack([
            np.vstack([(1 - p) * np.ones((n, 1)), [0]]), 
            np.vstack([p * np.eye(n), np.append(np.zeros(n - 1), 1)])
        ])
        
        # Cria vetor de Custo
        C = np.append(np.repeat(c, n), 0)
        
        # Cria vetor da funcao valor
        _V = np.zeros(n + 1)
        V = np.zeros(n + 1)
        
        for _ in range(_threshold):
            delta = np.tile(-V + C, (n + 1, 1)).T + np.tile(V, (n + 1, 1))
            pos = delta > 0
            neg = delta < 0
            _V = V.copy()
            V = V + alpha * np.sum((1 + k) * pos * T * delta + (1 - k) * neg * T * delta, axis=1)
            
            if max(abs(_V - V)) < _epsilon:
                # print(f'Convergiu com {_} passos')
                # print()
                break
            
        return {i: V[i] for i in range(len(V))}

    def mss_analytical_value_function(self, n: int, p: float, c: float, k: float):
        if n == 1:
            return self.osma_analytical_value_function(p, c, k)
        elif n == 2:
            def compute_V0(c, p, k, v1):
                # Numerator components
                numerator = c * (k - 2 * p * k + 1)
                denominator = p * (1 - k)
                
                # Compute V0
                V0 = (numerator / denominator) + v1
                return V0
            
            def compute_V1(c, p, k):
                # Numerator components
                term1 = p**2 * (1 - k)**2
                term2 = (1 - p) * (1 + k) * (1 + k + p - 3 * p * k)
                
                # Final numerator
                numerator = c * (term1 + term2)
                
                # Denominator
                denominator = p**2 * (1 - k)**2
                
                # Compute V1
                V1 = numerator / denominator
                return V1

            v1 = compute_V1(c, p, k)
            v0 = compute_V0(c, p, k, v1)
            
            return {0: v0, 1: v1}
        elif n == 3: # WIP - AINDA COM ERRO
            def calculate_V0(c, p, k):
                # Calculate V0 using the derived formula
                term1 = (1 - p) * (1 + k) * c
                term2 = (1 - k) * p
                
                V0 = term1 / term2 + c + calculate_V1(c, p, k)
                return V0

            def calculate_V1(c, p, k):
                # Calculate V1 using the derived formula
                term1 = (1 - p) * (1 + k) * (c + calculate_V0(c, p, k))
                return V1

            def calculate_V2(c, p, k):
                # Calculate V2 using the derived formula
                term1 = c
                term2 = ((1 - p) * (1 + k)) / (p * (1 - k))
                term3 = 2 * c + (c * (k - 2 * p * k + 1)) / (p * (1 - k)) + ((1 - p) * (1 + k) * c) / (p * (1 - k)) + ((1 - p) * (1 + k) * (k - 2 * p * k + 1)) / (p * (1 - k) ** 2)
                V2 = term1 + term2 * term3
                return V2
            
            v2 = calculate_V2(c, p, k)
            v1 = calculate_V1(c, p, k)
            v0 = calculate_V0(c, p, k)
            
            return {0: v0, 1: v1, 2: v2}
        else:
            res = {}
            for _n in range(n):
                res[_n] = np.nan
            return res

    def osma_value_function_range_probability(self, p: List[float], c: float, k: float, _verbose: bool = False):
        if _verbose:
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
            res[1][prob] = np.nan if res[1][prob][0] > 1e3 else res[1][prob]
                
        return res
    
    def mss_value_function_range_probability(self, n: List[int], p: List[float], c: float, k: float, alpha: float, gamma: float, _threshold: int, _epsilon: float, _verbose: bool = False, _validate_larger_values: bool = False):
        if _verbose:
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
                res[num_states][prob] = self.mss_value_function(num_states, prob, c, k, alpha, gamma, _threshold, _epsilon, _verbose)
                if _validate_larger_values: 
                    for state in res[num_states][prob].keys():
                        res[num_states][prob][state] = np.nan if res[num_states][prob][state] > 1e3 else res[num_states][prob][state]
                
        return res
    
    def mss_analytical_value_function_range_probability(self, n: List[int], p: List[float], c: float, k: float, _validate_larger_values: bool = False):
        res = {}
        
        for num_states in n:
            res[num_states] = {}
            for prob in p:
                prob = round(prob, 2)
                res[num_states][prob] = self.mss_analytical_value_function(num_states, prob, c, k)
                if _validate_larger_values: 
                    for state in res[num_states][prob].keys():
                        res[num_states][prob][state] = np.nan if res[num_states][prob][state] > 1e3 else res[num_states][prob][state]
                
        return res
    
    def diff_dict_values(self, dict1, dict2):
        # Ensure both dictionaries have the same keys
        assert dict1.keys() == dict2.keys(), "Dictionaries must have the same keys"
        
        # Sum the values for the same key in both dictionaries
        summed_values = {key: dict1[key] - dict2[key] for key in dict1}
        
        # Sum all the resulting values
        total_sum = sum(summed_values.values())
        
        return total_sum
    
    def mss_equivalent_cost_solver(self, n: float, p: float, cr: float, pr: float, k: float, alpha: float, gamma: float, _threshold: float, _epsilon: float, guess_EC: float = 0):
        def equation(EC, n, p, cr, pr, k, alpha, gamma, _threshold, _epsilon):
            if (n, pr, cr, k, alpha, gamma, _threshold, _epsilon) not in self.calculated_values.keys():
                self.calculated_values[(n, pr, cr, k, alpha, gamma, _threshold, _epsilon)] = self.mss_value_function(n, pr, cr, k, alpha, gamma, _threshold, _epsilon)
                
            reference_values = self.calculated_values[(n, pr, cr, k, alpha, gamma, _threshold, _epsilon)]
            
            if (n, p, EC[0], k, alpha, gamma, _threshold, _epsilon) not in self.calculated_values.keys():
                self.calculated_values[(n, p, EC[0], k, alpha, gamma, _threshold, _epsilon)] = self.mss_value_function(n, p, EC[0], k, alpha, gamma, _threshold, _epsilon)
            
            values = self.calculated_values[(n, p, EC[0], k, alpha, gamma, _threshold, _epsilon)]
            
            return self.diff_dict_values(reference_values, values)
        
        solution = fsolve(equation, guess_EC, args=(n, p, cr, pr, k, alpha, gamma, _threshold, _epsilon))
        
        return {0: solution[0]}
    
    def run_configuration_region(self, n: List[int], p: List[float], cr: float, pr: float, alpha: float, gamma: float, 
                                 _threshold: int=1e3, _epsilon: float=1e-3, guess_EC: float = 0):
        """
        Run the configuration region for the Exponential Utility Function.
            This method will run in two parts: (i) run the MSS value function for extreme positive value of lambda, and (ii) run the MSS value function for extreme negative value of lambda.
        """
        res = {}
        
        for num_states in tqdm(n, desc=" number states", position=0):
            res[num_states] = {}
            for prob in tqdm(p, desc=" probability", position=1, leave=False):
                prob = round(prob, 2)
                res[num_states][prob] = {}
                
                k_positive = 0.99
                k_negative = -0.99
                
                # Run the MSS value function for extreme positive value of lambda
                res[num_states][prob]['positive'] = self.mss_equivalent_cost_solver(num_states, prob, cr, pr, k_positive, 1 / (1 + abs(k_positive)), gamma, _threshold, _epsilon, guess_EC=0)
                
                # Run the MSS value function for extreme negative value of lambda
                res[num_states][prob]['negative'] = self.mss_equivalent_cost_solver(num_states, prob, cr, pr, k_negative, 1 / (1 + abs(k_negative)), gamma, _threshold, _epsilon, guess_EC=1)
                
        return res