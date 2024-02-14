
import numpy as np, random, copy, time
from collections import defaultdict

import ppgsi_mdp_risk.rl_utils.UtilFunctions as uf
from ppgsi_mdp_risk.models.Neutral_VI import Neutral_VI
from ppgsi_mdp_risk.rl_utils.VizTools import VizTools
from ppgsi_mdp_risk.services.ErrorMetrics import ErrorMetrics
from ppgsi_mdp_risk.services.ExponentialFunctions import ExponentialFunctions

class CVAR_PECVaR:
    def __init__(self, env, transition_probabilities, costs, 
                 alpha, max_iter=5, num_actions=4, epsilon=0.001, river_flow=None, discount_factor=0.99, QUIET=True) -> None:
        self.viz_tools = VizTools()
        self.env = env
        
        self._env_name = self.env._env_name
        self._river_flow = river_flow
        self._num_actions = num_actions
        self._alpha = alpha
        self._max_iter = max_iter
        self._threshold = 1000
        
        self._transition_probabilities = transition_probabilities
        self._costs = costs
        self._epsilon = epsilon
        self._discount_factor = discount_factor
        self._goal_state = env._goal_state
        
        # Inicializa Funcao Valor e Politica
        self.PI = self.env._build_PI0(initial_value=0)
        self.V = self.env._build_V0(initial_value=self._define_initial_value_V0())
        self.V_ANT = self.env._build_V0(initial_value=self._define_initial_value_V0())
        
        # Inicializa CVAR        
        self.CVaR0 = self.env._build_V0(initial_value=self._define_initial_value_V0())
        self.CVaR = self.env._build_V0(initial_value=self._define_initial_value_V0())
        
        self._first_run = True
        self._i = 0
        self.QUIET = QUIET
        
        # Guardando informação de convergencia
        self._inicia_historico_calculo()
        
        # Instanciando objetos
        self.EM = ErrorMetrics(self._epsilon)
        self.EF = ExponentialFunctions()
    
    def __repr__(self):
        if self._env_name == 'RiverProblem':
            self.viz_tools.visualize_V(self, self.V, self._grid_size, 4, self._goal_state, self._i, 
                                str_title=f'CVaR PEC - RSMDP - Alpha {self._alpha}')
            
            return f'RiverProblem - \n' + \
                f'Alpha: {self._alpha} \n' + \
                f'Epsilon: {self._epsilon} \n'
        else:
            return ''
    
    def _inicia_historico_calculo(self):
        self._hist_custo = {}
        self._hist_probabilty = {}
        self._hist_s0 = {}
        self._hist_G = {}
        self._policy_value = {}
        
        for S in self.V.keys():
            self._hist_custo[S] = {}
            self._hist_probabilty[S] = {}
            self._hist_s0[S] = {}
            self._hist_G[S] = {}
            self._policy_value[S] = {}
            for a in range(self._num_actions):
                self._hist_custo[S][a] = []
                self._hist_probabilty[S][a] = []
                self._hist_s0[S][a] = []
                self._hist_G[S][a] = []
                self._policy_value[S][a] = []
                
        return True
    
    def _define_initial_value_V0(self):
        if self._env_name == 'DrivingLicense': return 0
        elif self._env_name == 'RiverProblem': return 0
        else: return 0
    
    def _get_transition(self, S, a):
        if self._env_name == 'DrivingLicense': transition_matrix = self._transition_probabilities[S][a]
        elif self._env_name == 'RiverProblem': transition_matrix = self._transition_probabilities[a][S]
        else: transition_matrix = self._transition_probabilities[S][a]
        
        t = uf._get_values_from_dict(transition_matrix)
        return t    
        
    def _cost_function(self, S, action):
        if self._env_name == 'DrivingLicense' or self._env_name == 'SimpleMDP':
            if S == 'sG': return 0
        
        reward = self._costs[action]
        
        if self._env_name == 'RiverProblem':
            # Caso ele esteja na casa a direita do objetivo e a ação seja ir para esquerda
            if S == self._goal_state:
                reward = 0
                
        return reward
    
    def run_converge(self):
        while(self._first_run or (self.PI != self.PI_ANT)):
            print(f'Iteração: {self._i}', end='\r')
            
            V, PI = self.step()
            
            self._first_run = False
            self._i += 1
        
        return self._i, V, PI
    
    def step(self):
        self.policy_evaluation()
        self.policy_improvement()
        return self.CVaR0, self.PI
    
    def cvar_max_probability(self, CVaR, a):
        # In the case of Driving License, all the states 
        res = []
        
        for S in self.V.keys():            
            res.append(CVaR[S])
            
        return max(res)
    
    def policy_evaluation(self):
        # Policy Evaluation
        self._iteration = 0 # Inicia variavel iterativa
        V, CVaR, V_ANT = {}, {}, self.V.copy() # Inicia V e V (Anterior)
        
        while (self._iteration < self._max_iter): # For each value until convergence or MaxIterations
            for S in self.V.keys(): # For each state s in S
                # Define action
                a = self.PI[S]
                
                C = self._cost_function(S, a) # Cost for state S and action a
                T = self._get_transition(S, a) # Probability for state S and action a
                
                TV = T * uf._get_values_from_dict(self.V)
                V[S] = C + self._discount_factor * sum(TV)
                CVaR[S] = C + self._discount_factor * self.cvar_max_probability(self.CVaR0, a)
            
            self.CVaR0 = CVaR
            self.V = V.copy()
            self._iteration += 1
        
        return self.CVaR0
    
    def policy_improvement(self):
        PG, C, Vc, Ps, Psl = {}, {}, {}, {}, {}
        CVaR = {}
        for S in self.V.keys(): # For each state s in S
            # Initiate variables
            T = 0
            PG[0] = 0
            C[0] = 0
            Vc[0] = 0
            Ps[0] = 0
            Psl[0] = 0
            
            while ((1 -  PG[T]) < self.alpha):
                CVaR[S][(1 - PG[T])] = self.V[S] - Vc[T] * PG[T] / (1 - PG[T])
                T += 1
                
                for Sl in self.V.keys():
                    Psl[T] += self._get_transition(Sl, a) * Ps[T-1]
                    
                PG[T] = sum(Psl[T])
                C[T] = C[T-1] + self._discount_factor**(T-1)
                Vc[T] = (Vc[T-1] * PG[T-1] + C[T] * (PG[T] - PG[T-1])) / PG[T]
                
            for y in Y:
                T = min(t)
            
        self.PI_ANT = self.PI.copy()
        pass
        
    def calculate_value(self):
        
        
        # ! TO-DO:
        # * Criar dicionarios para parametros em cada execucao
        # * Entender variavel y e Y
        # * Retornar valores corretos para VCVaR
        # * Computar politica otima
                
        # After the CVaR interations, apply the following loop
        for S in self.V.keys(): # For each state s in S
            
            while (1 - Pt_G < self._alpha): # While the probability to the goal state is less than the defined alpha
                CVaR[S] = V[S] - Vt_C * Pt_G / (1 - Pt_G)
                Tt += 1
                
                for Sl in self.V.keys(): # For each state s in S
                    Pt_sl += self._get_transition(Sl, a) * Pt_s
                    
                Pt_G = sum(Pt_sl)
                Ct = Ct + self._discount_factor ** (Tt-1)
                Vt_C = (Vt_C * Pt_G + Ct * (Pt_G - Pt_G)) / Pt_G
            
            for y in Y:
                T = min(t)
                VCVaR[S,y] = ((1 - Pt_G) * CVaR[s,(1-Pt_G)] + (y - (1 - Pt_G)) * Ct) / y
                
            # Compute the optimal policy
            # for S in self.V.keys():
            #     for a in range(self._num_actions):
            #         q = self._get_transition(S, a) * uf._get_values_from_dict(self.V)
            #         C = self._cost_function(S, a)
            #         self.Qi[S][a] = np.exp(self._lambda * C) * sum(q)
                        
            #     self.PI[S] = min(self.Qi[S], key=self.Qi[S].get)
            
            return self._i, VCVaR, None
            
    # ! TODO
    def calculate_value_for_policy(self, Pi, vl_lambda):
        i = 0
        
        while True and i < self._threshold:
            new_V = {}
            for S in Pi.keys():
                a = Pi[S]
                
                q = self._get_transition(S, a) * uf._get_values_from_dict(self.V)
                C = self._cost_function(S, a)
                
                if S == self._goal_state:
                    new_V[S] = np.sign(self._lambda)
                else:
                    new_V[S] = np.exp(self._lambda * C) * sum(q)
            
            # print(f'Exp: {np.exp(vl_lambda * C)} / q: {sum(q)} ')
            i += 1
        
            if self.EM.relative_residual(uf._get_values_from_dict(self.V), uf._get_values_from_dict(new_V)):
                break
            
            self.V = copy.deepcopy(new_V)
            
        accumulate_cost = sum(uf._get_values_from_dict(self.V))            
        return accumulate_cost