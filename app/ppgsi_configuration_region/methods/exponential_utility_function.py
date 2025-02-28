
import numpy as np
from numpy import linalg as LA
from typing import List

from scipy.optimize import fsolve

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
                    C = c if S != 'sG' else 0
                    
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
        exp_lambda_c = np.exp(vl_lambda * c) * p
        
        term1 = ((exp_lambda_c)**n) * np.sign(vl_lambda)
        term2 = 1 - (1 - p) * np.exp(vl_lambda * c) * (1 - exp_lambda_c**n) / (1 - exp_lambda_c)
        
        result = term1 / term2 
        # return {0: np.exp(n * vl_lambda * c) * p**(n) * np.sign(vl_lambda) / (1 + sum([-np.exp(i * vl_lambda * c) * p**(i - 1) * (1-p) for i in range(1, n+1)]))}
        return {0: result}

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
        res = {}
        run_lambda_extreme = True if vl_lambda is None else False 
        
        for num_states in n:
            res[num_states] = {}
            for prob in p:
                prob = round(prob, 2)
                
                if run_lambda_extreme:
                    vl_lambda = LambdaExtreme().solver_lambda_extreme_mss(num_states, prob, c)[0] - 0.001
                    # print(num_states, prob, vl_lambda)
                
                res[num_states][prob] = self.mss_value_function(num_states, prob, c, vl_lambda, _threshold, _epsilon, _alpha, _quiet)
                
                for state in res[num_states][prob].keys():
                    res[num_states][prob][state] = np.nan if res[num_states][prob][state] > 1e3 else res[num_states][prob][state]
                
        return res
    
    def mss_analytical_value_function_range_probability(self, n: List[int], p: List[float], c: float, vl_lambda: float):
        res = {}
        run_lambda_extreme = True if vl_lambda is None else False 
        
        for num_states in n:
            res[num_states] = {}
            for prob in p:
                prob = round(prob, 2)
                
                if run_lambda_extreme:
                    vl_lambda = LambdaExtreme().solver_lambda_extreme_mss(num_states, prob, c)[0] - 0.001
                
                res[num_states][prob] = self.mss_analytical_value_function(num_states, prob, c, vl_lambda)
                
                for state in res[num_states][prob].keys():
                    res[num_states][prob][state] = np.nan if (res[num_states][prob][state] > 1e3) or (res[num_states][prob][state] < 0) else res[num_states][prob][state]
                
        return res
        
    def diff_dict_values(self, dict1, dict2):
        # Ensure both dictionaries have the same keys
        assert dict1.keys() == dict2.keys(), "Dictionaries must have the same keys"
        
        # Sum the values for the same key in both dictionaries
        summed_values = {key: dict1[key] - dict2[key] for key in dict1}
        
        # Sum all the resulting values
        total_sum = sum(summed_values.values())
        
        return total_sum

    def mss_equivalent_cost_solver(self, n: float, p: float, cr: float, pr: float, vl_lambda: float, _threshold: float, _epsilon: float, _alpha: float = 1, guess_EC: float = 0):
        def equation(EC, n, p, cr, pr, vl_lambda, _threshold, _epsilon, _alpha):
            reference_values = self.mss_value_function(n, pr, cr, vl_lambda, _threshold, _epsilon, _alpha)
            values = self.mss_value_function(n, p, EC[0], vl_lambda, _threshold, _epsilon, _alpha)
            
            return self.diff_dict_values(reference_values, values)
        
        solution = fsolve(equation, guess_EC, args=(n, p, cr, pr, vl_lambda, _threshold, _epsilon, _alpha))
        
        # print(f"cr: {cr} | pr: {pr} | lambda: {vl_lambda} | n: {n} | p: {p} | EC: {solution}")
        return {0: solution[0]}

    def mss_analytical_equivalent_cost_solver(self, n: float, p: float, cr: float, pr: float, vl_lambda: float, guess_EC: float = 0):
        def equation(EC, n, p, cr, pr, vl_lambda):
            reference_values = self.mss_analytical_value_function(n, pr, cr, vl_lambda)
            values = self.mss_analytical_value_function(n, p, EC[0], vl_lambda)
            
            return self.diff_dict_values(reference_values, values)
        
        solution = fsolve(equation, guess_EC, args=(n, p, cr, pr, vl_lambda))
        
        # print(f"cr: {cr} | pr: {pr} | lambda: {vl_lambda} | n: {n} | p: {p} | EC: {solution}")
        return {0: solution[0]}
           
    def run_configuration_region(self, n: List[int], p: List[float], cr: float, pr: float, analytical: bool=False, _threshold: int=1e3, _epsilon: float=1e-3, _alpha: float=1, guess_EC: float = 0, _quiet: bool=True):
        """
        Run the configuration region for the Exponential Utility Function.
            This method will run in two parts: (i) run the MSS value function for extreme positive value of lambda, and (ii) run the MSS value function for extreme negative value of lambda.
        """
        res = {}
        
        for num_states in n:
            res[num_states] = {}
            for prob in p:
                prob = round(prob, 2)
                res[num_states][prob] = {}
                
                vl_lambda_extreme_reference = LambdaExtreme().solver_lambda_extreme_mss(num_states, pr, cr)[0]
                vl_lambda_extreme = LambdaExtreme().solver_lambda_extreme_mss(num_states, prob, cr)[0]
                vl_lambda_extreme = min([vl_lambda_extreme, vl_lambda_extreme_reference])
                
                if analytical:
                    # Run the MSS value function for extreme positive value of lambda
                    res[num_states][prob]['positive'] = self.mss_analytical_equivalent_cost_solver(num_states, prob, cr, pr, vl_lambda_extreme - 0.01, guess_EC=1 if prob >= 0.5 else 0)
                    
                    # Run the MSS value function for extreme negative value of lambda
                    res[num_states][prob]['negative'] = self.mss_analytical_equivalent_cost_solver(num_states, prob, cr, pr, -1e5, guess_EC=1)
                else:
                    # Run the MSS value function for extreme positive value of lambda
                    res[num_states][prob]['positive'] = self.mss_equivalent_cost_solver(num_states, prob, cr, pr, vl_lambda_extreme - 0.01, _threshold, _epsilon, _alpha, _quiet)
                    
                    # Run the MSS value function for extreme negative value of lambda
                    res[num_states][prob]['negative'] = self.mss_equivalent_cost_solver(num_states, prob, cr, pr, -1e5, _threshold, _epsilon, _alpha, _quiet)
                
        return res

    def run_configuration_region_number_states(self, n: List[int], p: List[float], c: float, pr: float, nr: float, analytical: bool=False, _threshold: int=1e3, _epsilon: float=1e-3, _alpha: float=1, _quiet: bool=True):
        if analytical:
            # Positive Values
            positive_values = self.mss_analytical_value_function_range_probability(n, p, c, None)
            
            # Negative Values
            negative_values = self.mss_analytical_value_function_range_probability(n, p, c, -1e5)
        else:
            # Positive Values
            positive_values = self.mss_value_function_range_probability(n, p, c, None, _threshold, _epsilon)
            
            # Negative Values
            negative_values = self.mss_value_function_range_probability(n, p, c, -1e5, _threshold, _epsilon)
        
        positive_values_for_each_probability = {}
        negative_values_for_each_probability = {}
        
        for num_states in positive_values.keys():
            for prob in positive_values[num_states].keys():
                if prob not in positive_values_for_each_probability.keys(): positive_values_for_each_probability[prob] = np.array([])
                positive_values_for_each_probability[prob] = np.append(positive_values_for_each_probability[prob], positive_values[num_states][prob][0])
                
        for num_states in negative_values.keys():
            for prob in negative_values[num_states].keys():
                if prob not in negative_values_for_each_probability.keys(): negative_values_for_each_probability[prob] = np.array([])
                negative_values_for_each_probability[prob] = np.append(negative_values_for_each_probability[prob], negative_values[num_states][prob][0])
                
        positive_reference_value = positive_values[nr][pr][0]
        negative_reference_value = negative_values[nr][pr][0]
        
        pos = np.array([])
        for prob in p:
            prob = prob.round(2)
            try:
                print(positive_values_for_each_probability[prob][positive_values_for_each_probability[prob] - positive_reference_value < 0])
                max_number = np.argmax(positive_values_for_each_probability[prob][positive_values_for_each_probability[prob] - positive_reference_value < 0]) + 1
            except:
                max_number = 1
            pos = np.append(pos, max_number)
            
        neg = np.array([])
        for prob in p:
            prob = prob.round(2)
            try:
                max_number = np.argmax(negative_values_for_each_probability[prob][negative_values_for_each_probability[prob] - negative_reference_value < 0]) + 1
            except:
                max_number = 1
            neg = np.append(neg, max_number)
            
        res = {}
        res['positive'] = pos
        res['negative'] = neg
            
        return res
    
    def run_solver_for_number_states(self, n: List[int], p: List[float], c: float, pr: float, nr: float, vl_lambda: float, analytical: bool=False, _threshold: int=1e3, _epsilon: float=1e-3, _alpha: float=1, _quiet: bool=True):
        # Negative Values
        values = self.mss_value_function_range_probability(n, p, c, vl_lambda, _threshold, _epsilon)
        values_for_each_probability = {}
        
        for num_states in values.keys():
            for prob in values[num_states].keys():
                if prob not in values_for_each_probability.keys(): values_for_each_probability[prob] = np.array([])
                values_for_each_probability[prob] = np.append(values_for_each_probability[prob], values[num_states][prob][0])
                
        reference_value = values[nr][pr][0]
        
        v = np.array([])
        for prob in p:
            prob = prob.round(2)
            try:
                max_number = np.argmax(values_for_each_probability[prob][values_for_each_probability[prob] - reference_value < 0]) + 1
            except:
                max_number = 1
            v = np.append(v, max_number)
            
        res = {}
        res['positive'] = v
            
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
    
    def solver_lambda_extreme_mss(self, n, p, c, lambda_guess=0.1):
        """
        Solves for lambda using numerical methods.

        Parameters:
        - p: Probability parameter (0 < p < 1)
        - c: Constant multiplier
        - n: Integer exponent
        - lambda_guess: Initial guess for lambda

        Returns:
        - Approximate solution for lambda.
        """
        def equation(lambda_val, p, c, n):
            """
            Defines the equation to solve for lambda.
            """
            term = 1 - (1 - p) * np.exp(lambda_val * c) * (1 - (np.exp(lambda_val * c) * p)**n) / (1 - (np.exp(lambda_val * c) * p))
            return term
        
        solution = fsolve(equation, lambda_guess, args=(p, c, n))
        return {0: solution[0]}
    
    def find_numerical_lambda_extreme_range_probability(self, n, p, c):
        res = {}
        
        for num_states in n:
            res[num_states] = {}
            for prob in p:
                prob = round(prob, 2)
                res[num_states][prob] = self.solver_lambda_extreme_mss(num_states, prob, c)
                
        return res