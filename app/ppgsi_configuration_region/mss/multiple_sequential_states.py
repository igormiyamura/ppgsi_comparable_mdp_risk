import random

class MultipleSequentialStates:
    """
    A class representing a Markov Decision Process (MDP) with multiple sequential states.

    Attributes:
        _env_name (str): Name of the environment.
        _num_states (int): Number of sequential states.
        _num_actions (int): Number of actions available.
        _goal_state (str): Identifier for the goal state.
        _actions (list): List of possible actions.
        _states (list): List of states, including the goal state.
        _probability (float): Transition probability to the next state.
        _cost (int): Default cost associated with transitions.
    """
    
    def __init__(self, num_states, probability=0.8, cost=1) -> None:
        """
        Initialize the MDP environment.

        Args:
            num_states (int): Number of sequential states.
            probability (float): Probability of transitioning to the next state. Default is 0.8.
            cost (int): Cost of transitions. Default is 1.
        """
        self._env_name = 'MultipleSequentialStates'
        
        self._num_states = num_states
        self._num_actions = 1
        self._goal_state = 'sG'
        
        self._actions = [a for a in range(0, 1)]
        self._states = [s for s in range(0, self._num_states)]
        self._states.append(self._goal_state)
        
        self._probability = probability
        self._cost = cost
        
    def _verify_sum_probabilities(self, transition_probabilities):
        """
        Verify that the sum of transition probabilities for each action-state pair equals 1.

        Args:
            transition_probabilities (dict): Transition probability dictionary.

        Returns:
            tuple: A boolean indicating success, and a dictionary of verification results.
        """
        is_ok, dict_verification = True, {}
        for action in transition_probabilities.keys():
            for state in transition_probabilities[action].keys():
                dict_verification[(action, state)] = True if sum([v[1] for v in transition_probabilities[action][state].items()]) == 1 else False

                if not dict_verification[(action, state)]:
                    print(action, state, [v[1] for v in transition_probabilities[action][state].items()])
                    is_ok = False
                        
        return is_ok, dict_verification
    
    def _build_V0(self, initial_value=0):
        """
        Build an initial value function (V).

        Args:
            initial_value (int): Initial value for all states. Default is 0.

        Returns:
            dict: Value function initialized for all states.
        """
        V = {}
        
        for s in self._states:
            V[s] = initial_value
            
        return V
    
    def _build_PI0(self, initial_value=-1):
        """
        Build an initial policy (PI).

        Args:
            initial_value (int): Initial policy value for all states. Default is -1.

        Returns:
            dict: Policy initialized for all states.
        """
        PI = {}
        
        for s in self._states:
            PI[s] = initial_value
            
        return PI
    
    def _build_Q0(self, initial_value=0):
        """
        Build an initial action-value function (Q).

        Args:
            initial_value (int): Initial value for all state-action pairs. Default is 0.

        Returns:
            dict: Action-value function initialized for all states and actions.
        """
        Q0 = {}
        for s in self._states:
            Q0[s] = {}
            for a in self._actions:
                Q0[s][a] = 0
        return Q0
    
    def build_default_states_action_transition_dictionary(self, states, actions):
        """
        Build a default transition dictionary for all state-action pairs.

        Args:
            states (list): List of states.
            actions (int): Number of actions.

        Returns:
            dict: Nested dictionary representing default transitions.
        """
        res = {}
        
        for s in states:
            res[s] = {}
            for a in range(0, actions):
                res[s][a] = {}
                for s_next in states:
                    res[s][a][s_next] = 0
        
        return res
        
    def build_transition_probabilities(self):
        """
        Build the transition probability matrix for the environment.

        Returns:
            dict: Transition probabilities for all state-action pairs.
        """
        T = self.build_default_states_action_transition_dictionary(self._states, self._num_actions)
        
        for s in T.keys():
            for a in T[s].keys():
                if s == 'sG':
                    T[s][a]['sG'] = 1 # If in the goal state, stay with probability 1
                else:
                    if s + 1 >= self._num_states:
                        T[s][a]['sG'] = self._probability # Transition to the goal state
                        T[s][a][0] = 1 - self._probability # Stay in the same state
                    else:
                        T[s][a][s + 1] = self._probability # Transition to the next state
                        T[s][a][0] = 1 - self._probability # Stay in the same state
        
        self._verify_sum_probabilities(T)
        
        return T
