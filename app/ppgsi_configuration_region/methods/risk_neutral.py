
import numpy as np

from .value_function_calculator import ValueFunctionCalculator

class RiskNeutral(ValueFunctionCalculator):
    def __init__(self, _verbose=False) -> None:
        self._verbose = _verbose
    
    def osma_analytical_value_function(self, *args, **kwargs):
        pass

    def mss_value_function(self, c, p, n, num_simulations=100000):
        costs = []
        for _ in range(num_simulations):
            ActualState = {}
            for j in range(n): ActualState[j] = False
            ActualState[0] = True
            ActualState[n] = False

            _acc = 0
            _cost = 0

            while True:
                _actual_probability = np.random.rand()
                _cost += c
                if self._verbose: print(f'State: {ActualState}, Probability: {_actual_probability}, Cost: {_cost}')
                
                ActualState[_acc] = False
                if _actual_probability >= p:
                    _acc += 1
                    ActualState[_acc] = True
                else:
                    _acc = 0
                    ActualState[_acc] = True

                if ActualState[n] == True:
                    break
                
            costs.append(_cost)
        return np.mean(costs), costs  
    
    def mss_analytical_value_function(self, c, p, n):
        return c * sum([p**i for i in range(0, n)]) / p**(n)
    
    def osma_value_function_range_probability(self, *args, **kwargs):
        pass
    
    def mss_value_function_range_probability(self, *args, **kwargs):
        pass