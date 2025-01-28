
import numpy as np
from numpy import linalg as LA
from typing import List

from ..residuals.error_metrics import ErrorMetrics
from ..mss.multiple_sequential_states import MultipleSequentialStates
from .value_function_calculator import ValueFunctionCalculator

class ExponentialUtilityFunction(ValueFunctionCalculator):
    def __init__(self) -> None:
        pass
    
    def mss_value_function(self, n: int, p: float, c: float, vl_lambda: float, _threshold: int, _epsilon: float, _alpha: float=1, _quiet: bool=True):
        self.EM = ErrorMetrics(_epsilon)
        iterator = 0

        # Define Multiple Sequential States
        MSS = MultipleSequentialStates(num_states=n, probability=p, cost=c)
        mss_value_function = MSS._build_V0()
        mss_transitions = MSS.build_transition_probabilities()

        if not _quiet: display(mss_value_function)
        if not _quiet: display(mss_transitions)

        # Cria variaveis Value Function e Trasition
        V = mss_value_function.copy()
        T = mss_transitions.copy()
        while True and iterator < _threshold:
            new_V = {}
            for S in mss_transitions.keys():
                bellman_results = []
                for a in mss_transitions[S].keys():
                    if not _quiet: print(f'State: {S}, Action: {a}, {iterator} < {_threshold}', end='\r')
                    v = np.array(list(V.values()))
                    t = np.array(list(T[S][a].values()))
                    
                    TV = t * v
                    C = 1 if S != 'sG' else 0
                    
                    bellman = np.exp(C * vl_lambda) * _alpha * sum(TV)
                    bellman_results.append(bellman)
                    
                if S == MSS._goal_state:
                    new_V[S] = np.sign(vl_lambda)
                else:
                    new_V[S] = min(bellman_results)
                    
            iterator += 1
            
            if not _quiet: print('V:', new_V)
            
            if self.EM.relative_residual(np.array(list(V.values())), np.array(list(new_V.values()))):
                V = new_V.copy()
                break
            
            V = new_V.copy()
        return V
        
    def osma_analytical_value_function(self, p: float, c: float, vl_lambda: float):
        return {0: (np.sign(vl_lambda) * np.exp(vl_lambda * c) * p) / (1 - np.exp(vl_lambda * c) * (1 - p))}
        
    def mss_analytical_value_function(self, n: int, p: float, c: float, vl_lambda: float):
        return np.exp(n * vl_lambda * c) * p**(n) * np.sign(vl_lambda) / (1 + sum([-np.exp(i * vl_lambda * c) * p**(i - 1) * (1-p) for i in range(1, n+1)]))

    def osma_value_function_range_probability(self, p: List[float], c: float, vl_lambda: float):
        print(f"""
              Calculando valores para os seguintes parâmetros:
                p: {p} | 
                c: {c} | 
                lambda: {vl_lambda} |
              """)
        
        res = {}
        res[1] = {}
        
        for prob in p:
            prob = round(prob, 2)
            res[1][prob] = self.osma_analytical_value_function(prob, c, vl_lambda)
                
        return res

    def mss_value_function_range_probability(self, n: List[int], p: List[float], c: float, vl_lambda: float, _threshold: int, _epsilon: float, _alpha: float=1, _quiet: bool=True):
        print(f"""
              Calculando valores para os seguintes parâmetros:
                n: {[v for v in n]} | 
                p: {[round(v, 2) for v in p]} | 
                c: {c} | 
                lambda: {vl_lambda} |
                threshold: {_threshold} | 
                epsilon: {_epsilon} | 
                alpha: {_alpha} |
              """)
        
        res = {}
        
        for num_states in n:
            res[num_states] = {}
            for prob in p:
                prob = round(prob, 2)
                res[num_states][prob] = self.mss_value_function(num_states, prob, c, vl_lambda, _threshold, _epsilon, _alpha, _quiet)
                
                for state in res[num_states][prob].keys():
                    res[num_states][prob][state] = np.nan if res[num_states][prob][state] > 1e3 else res[num_states][prob][state]
                
        return res
        

class LambdaExtreme:
    def __init__(self) -> None:
        pass
    
    def create_vector_D(self, S, c):
        D = np.zeros((len(S), len(S)))

        for row in range(len(D)):
            D[row][row] = np.exp(c)
            
        return D

    def create_vector_T(self, S, mss_transitions):
        T = np.zeros((len(S), len(S)))
        _row, _column = 0, 0
        for row in range(len(T)):
            _row = 'sG' if row == len(T)-1 else row
            for column in range(len(T[row])):
                _column = 'sG' if column == len(T)-1 else column
                T[row][column] = mss_transitions[_row][0][_column] if _column != 'sG' else 0
            
        return T

    def spectral_radius(self, X, Y):
        return max(abs(LA.eig(X * Y)[0]))

    def find_lambda_extreme(self, n, p, c, vl_lambda, epsilon, beta, _quiet=True):
        MSS = MultipleSequentialStates(num_states=n, probability=p, cost=c)
        
        mss_value_function = MSS._build_V0()
        mss_transitions = MSS.build_transition_probabilities()

        S = mss_transitions.keys()
        D = self.create_vector_D(S, c) ** vl_lambda
        T = self.create_vector_T(S, mss_transitions)

        if not _quiet: print(f'Initial Spectral Radius: {self.spectral_radius(D, T)}')

        pi0 = MSS._build_PI0(initial_value=0)
        pi = MSS._build_PI0(initial_value=-1)
        i = 0

        while pi != pi0:
            while self.spectral_radius(D, T) <= (1 - beta):
                # While
                # -----------
                if not _quiet: print(f'While: {self.spectral_radius(D, T)} >= {(1 - beta)}')
                
                # Step
                # ----
                step = (np.log(1 - epsilon) - np.log(self.spectral_radius(D, T))) / c
                vl_lambda = vl_lambda + step
                if not _quiet: print(f'Lambda Extreme: {vl_lambda} > Step: {step}')
                
                D = self.create_vector_D(S, c) ** vl_lambda

                if not _quiet: print(f'Iteration: {i}')
                pi = MSS._build_PI0(initial_value=0)
                i += 1
                
        return {0: vl_lambda}
    
    def find_lambda_extreme_range_probability(self, n, p, c, vl_lambda, epsilon, beta):
        res = {}
        
        for num_states in n:
            res[num_states] = {}
            for prob in p:
                prob = round(prob, 2)
                res[num_states][prob] = self.find_lambda_extreme(num_states, prob, c, vl_lambda, epsilon, beta)
                
        return res
                