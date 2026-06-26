from base.game import AgentID, ObsType
from numpy import ndarray
from gymnasium.spaces import Discrete, Text, Dict, Tuple
from pettingzoo.utils import agent_selector
from pettingzoo.classic import leduc_holdem_v4 as leduc
from base.game import AlternatingGame, AgentID, ActionType
import numpy as np
from functools import reduce

import warnings
warnings.filterwarnings("ignore")

class Leduc(AlternatingGame):

    def __init__(self, render_mode=''):
        super().__init__()
        self.env = leduc.raw_env(render_mode=render_mode)
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces
        self.action_space = self.env.action_space
        self.agents = self.env.agents
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self.render_mode = render_mode
        self._hist = ''
        self._moves = ['c', 'r', 'f', 'k']

    def _update(self):
        self.rewards = self.env.rewards
        self.terminations = self.env.terminations
        self.truncations = self.env.truncations
        self.infos = self.env.infos
        self.agent_selection = self.env.agent_selection

    def observe(self, agent: AgentID) -> ObsType:
        state = self.env.env.game.get_state(self.env._name_to_int(agent))
        hand = state['hand'][1]
        public_card = '#' if state['public_card'] is None else state['public_card'][1]
        chips = '_'.join([str(x) for x in state['all_chips']])
        obs = hand + '_' + public_card + '_' + chips + '_' + self._hist
        return obs
    
    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self.env.reset(seed, options)
        self._update()
        self._hist = str(self.env._name_to_int(self.agent_selection))
    
    def render(self) -> ndarray | str | list | None:
        return self.env.render()
    
    def step(self, action: ActionType) -> None:
        self._hist += self._moves[action] 
        self.env.step(action)
        self._update()

    def available_actions(self):
        return list(self.env.next_legal_moves)
    
    def close(self):
        self.env.close()