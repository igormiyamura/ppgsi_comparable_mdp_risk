
import numpy as np, random
import time

from ppgsi_mdp_risk.rl_utils.VizTools import VizTools

class Neutral_PI:
    def __init__(self, env, transition_probabilities, costs, 
                 num_actions=4, discount_factor=0.95, epsilon=0.001, river_flow=None) -> None:
        self.env = env
        self.viz_tools = VizTools()
        
        self._env_name = self.env._env_name
        
        self._river_flow = river_flow
        self._num_actions = num_actions
        self._goal_state = self.env._goal_state
        
        self._transition_probabilities = transition_probabilities
        self._costs = costs
        
        self._discount_factor = discount_factor
        self._epsilon = epsilon
        self.V = self.env._build_V0(initial_value=0)
        self.PI = self.env._build_PI0(initial_value=0)
        self._first_run = True
        self._i = 0
    
    def __repr__(self):
        if self._env_name == 'RiverProblem':
            self.viz_tools.visualize_V(self, self.V, self._grid_size, 4, self._goal_state, self._i, 
                                str_title=f'Policy Iteration')
            
            return f'RiverProblem - \n' + \
                f'Discount Factor: {self._discount_factor} \n' + \
                f'Epsilon: {self._epsilon} \n'
        return None
        
    def _get_random_action(self):
        return int(random.choice([i for i in range(0, self._num_actions)]))
        
    def _reward_function(self, S, action):
        if self._env_name == 'DrivingLicense':
            if S == 'sG': return 0
        
        reward = self._costs[action]
        
        if self._env_name == 'RiverProblem':
            # Caso ele esteja na casa a direita do objetivo e a ação seja ir para esquerda
            if S == self._goal_state:
                reward = 0
                
        return reward
    
    def _get_transition(self, S, a):
        if self._env_name == 'DrivingLicense': transition_matrix = self._transition_probabilities[S][a]
        elif self._env_name == 'RiverProblem': transition_matrix = self._transition_probabilities[a][S]
        t = np.array([v[1] for v in transition_matrix.items()])
        return t
    
    def _get_V(self):
        V = np.array([v[1] for v in self.V.items()])
        return V
    
    def _next_state(self, state, action):
        x, y = state
        if action == 0:   # up
            x = max(x - 1, 0)
        elif action == 1: # down
            x = min(x + 1, self._rows - 1)
        elif action == 2: # left
            y = max(y - 1, 0)
        elif action == 3: # right
            y = min(y + 1, self._cols - 1)
        return (x, y)
        
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
        return self.V, self.PI
    
    def policy_evaluation(self):
        V, V_ANT, i = {}, self.V.copy(), 0
        
        while(i == 0 or \
            np.max(np.abs( np.subtract(list(self.V.values()), list(V_ANT.values())) )) > 2 * self._epsilon):
            
            V_ANT = self.V.copy()
            for S in self.V.keys():
                a = self.PI[S]
                
                bellman = self._reward_function(S, a) + self._discount_factor * \
                    (self._get_transition(S, a) * self._get_V()).sum()
                
                V[S] = bellman
            
            self.V = V.copy()
            i += 1
        return self.V
    
    def policy_improvement(self):
        self.PI_ANT = self.PI.copy()
        
        pi_improved = {}
        for S in self.V.keys():
            bellman = {}
            for a in range(0, self._num_actions):
                b = self._reward_function(S, a) + self._discount_factor * \
                    (self._get_transition(S, a) * self._get_V()).sum()
                    
                bellman[a] = b
                
            pi_improved[S] = min(bellman, key=bellman.get)
        
        self.PI = pi_improved
        return self.PI
    
    