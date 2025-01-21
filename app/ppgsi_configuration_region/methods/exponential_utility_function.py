
import numpy as np
from numpy import linalg as LA

from ..residuals.error_metrics import ErrorMetrics
from ..mss.multiple_sequential_states import MultipleSequentialStates

class ExponentialUtilityFunction:
    def __init__(self, epsilon) -> None:
        self.epsilon = epsilon
        self.EM = ErrorMetrics(self.epsilon)
    
    def exponential_utility_method(self, n: int, p: float, c: float, vl_lambda: float, _threshold: int, _epsilon: float, _alpha: float=1, _quiet: bool=True):
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
        
    def exponential_utility_method_analitico(self, n: int, p: float, c: float, vl_lambda: float):
        return np.exp(n * vl_lambda * c) * p**(n) * np.sign(vl_lambda) / (1 + sum([-np.exp(i * vl_lambda * c) * p**(i - 1) * (1-p) for i in range(1, n+1)]))

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

    def find_lambda_extreme(self, n, p, c, vl_lambda, epsilon, beta):
        MSS = MultipleSequentialStates(num_states=n, probability=p, cost=c)
        
        mss_value_function = MSS._build_V0()
        mss_transitions = MSS.build_transition_probabilities()

        S = mss_transitions.keys()
        D = self.create_vector_D(S, c) ** vl_lambda
        T = self.create_vector_T(S, mss_transitions)

        print(f'Initial Spectral Radius: {self.spectral_radius(D, T)}')

        pi0 = MSS._build_PI0(initial_value=0)
        pi = MSS._build_PI0(initial_value=-1)
        i = 0

        while pi != pi0:
            while self.spectral_radius(D, T) <= (1 - beta):
                # Print While
                # -----------
                print(f'While: {self.spectral_radius(D, T)} >= {(1 - beta)}')
                
                # Step
                # ----
                step = (np.log(1 - epsilon) - np.log(self.spectral_radius(D, T))) / c
                vl_lambda = vl_lambda + step
                print(f'Lambda Extreme: {vl_lambda} > Step: {step}')
                
                D = self.create_vector_D(S, c) ** vl_lambda

                print(f'Iteration: {i}')
                pi = MSS._build_PI0(initial_value=0)
                i += 1
                
        return vl_lambda