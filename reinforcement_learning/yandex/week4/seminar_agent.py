from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class QLearningAgent:
    def __init__(self, state_dim, n_actions, gamma):

        self.n_actions = n_actions
        self.gamma = gamma
        
        network = nn.Sequential()
        network.add_module('dense1', nn.Linear(state_dim[0], 64))
        network.add_module('relu1', nn.ReLU())
        network.add_module('dense2',  nn.Linear(64, 64))
        network.add_module('relu2', nn.ReLU())
        network.add_module('dense3',  nn.Linear(64, n_actions))
        
        self.network = network       
        

    def get_action(self, state, epsilon):
        """
        sample actions with epsilon-greedy policy
        recap: with p = epsilon pick random action, else pick action with highest Q(s,a)
        """
        state = torch.tensor(state[None], dtype=torch.float32)
        q_values = self.network(state).detach().numpy()

        chosen_action = 0
        if epsilon > np.random.uniform(0, 1):
                chosen_action = np.random.randint(0, self.n_actions)
        #                 chosen_action = np.random.choice(range(self.n_actions))
        else:
            chosen_action = np.argmax(q_values) 

        return int(chosen_action)

    def compute_td_loss(self, states, actions, rewards, next_states, is_done, check_shapes=False):
        """ Compute td loss using torch operations only. Use the formula above. """
        states = torch.tensor(states, dtype=torch.float32)    # shape: [batch_size, state_size]
        actions = torch.tensor(actions, dtype=torch.long)    # shape: [batch_size]
        rewards = torch.tensor(rewards, dtype=torch.float32)  # shape: [batch_size]
        
        # shape: [batch_size, state_size]
        next_states = torch.tensor(next_states, dtype=torch.float32)
        is_done = torch.tensor(is_done, dtype=torch.uint8)  # shape: [batch_size]

        # get q-values for all actions in current states
        predicted_qvalues = self.network(states)

        # select q-values for chosen actions
        predicted_qvalues_for_actions = predicted_qvalues[
          range(states.shape[0]), actions
        ]

        # compute q-values for all actions in next states
        predicted_next_qvalues = self.network(next_states)

        # compute V*(next_states) using predicted next q-values
        next_state_values = predicted_next_qvalues.max(dim=-1)[0]
        # the same     next_state_values = torch.max(predicted_next_qvalues, dim=-1)[0]

        # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
        target_qvalues_for_actions = rewards + self.gamma * next_state_values

        # at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
        target_qvalues_for_actions = torch.where(
            is_done, rewards, target_qvalues_for_actions)

        # mean squared error loss to minimize
        loss = torch.mean((predicted_qvalues_for_actions -
                           target_qvalues_for_actions.detach()) ** 2)

        return loss
    