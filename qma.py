import numpy as np
from enum import IntEnum

class QActions(IntEnum):
    BACKOFF = 0
    CCA = 1
    SEND = 2

class QMA:
    def __init__(self, nodes, slots, alpha=0.5, gamma=0.9, xi=2.0, rho=0.1):
        self.nodes = nodes
        self.slots = slots
        self.alpha = alpha
        self.gamma = gamma
        self.xi = xi
        self.rho = rho
        self.qtable = np.full((self.nodes, self.slots, len(QActions)), -10.0)
        self.policies = np.full((self.nodes, self.slots), QActions.BACKOFF)

    def reset(self):
        self.qtable = np.full((self.nodes, self.slots, len(QActions)), -10.0)
        self.policies = np.full((self.nodes, self.slots), QActions.BACKOFF)

    def get_next_timestep(self, slot):
        assert(0 <= slot < self.slots)
        actions, random_actions = self._select_actions(slot)
        rewards = self._calculate_rewards(actions)
        self._update_qtables(slot, actions, rewards)
        return self.qtable, actions, random_actions, rewards

    def _select_actions(self, slot):
        actions = np.copy(self.policies[:,slot])
        random_actions = [False for node in range(self.nodes)] 
        for action_idx in range(len(actions)):
            if np.random.random() < self.rho:
                actions[action_idx] = np.random.choice(list(QActions))
                random_actions[action_idx] = True
        return actions, random_actions 

    def _calculate_rewards(self, actions):
        assert(len(actions) == self.nodes)
        rewards = np.array([-999 for action in actions])
        for action_idx, action in enumerate(actions):
            if action == QActions.BACKOFF:
                if not self._is_collision(actions, QActions.CCA) and not self._is_collision(actions, QActions.SEND) and np.count_nonzero(actions == QActions.BACKOFF) != len(actions):
                    rewards[action_idx] = 2
                else:
                    rewards[action_idx] = 0
            elif action == QActions.CCA:
                if self._is_cca_success(actions):
                    if not self._is_collision(actions, QActions.CCA):
                        rewards[action_idx] = 3
                    else:
                        rewards[action_idx] = -2
                else:
                    rewards[action_idx] = 1

            elif action == QActions.SEND:
                if self._is_collision(actions, QActions.SEND):
                    rewards[action_idx] = -3
                else:
                    rewards[action_idx] = 4
        return rewards

    def _is_collision(self, actions, ownAction):
        return np.count_nonzero(actions == ownAction) > 1

    def _is_cca_success(self, actions):
        return np.count_nonzero(actions == QActions.SEND) == 0

    def _update_qtables(self, slot, actions, rewards):
        assert(len(rewards) == self.nodes and len(actions) == self.nodes and 0 <= slot < self.slots)
        for node,action in enumerate(actions):
            other_q = (1-self.alpha)*self.qtable[node,slot,action] + self.alpha*(rewards[node] + self.gamma*np.max(self.qtable[node,(slot+1)%self.slots,:]))
            self.qtable[node,slot,action] = np.max([self.qtable[node,slot,action]-self.xi, other_q])
            if np.max(self.qtable[node,slot,:]) != self.qtable[node,slot,self.policies[node,slot]]:
                self.policies[node,slot] = np.argmax(self.qtable[node,slot,:])   
