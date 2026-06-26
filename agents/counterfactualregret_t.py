import numpy as np
from numpy import ndarray
from base.game import AlternatingGame, AgentID, ObsType
from base.agent import Agent

class Node():

    def __init__(self, game: AlternatingGame, obs: ObsType) -> None:
        self.game = game
        self.agent = game.agent_selection
        self.obs = obs
        self.num_actions = self.game.num_actions(self.agent)
        self.cum_regrets = np.zeros(self.num_actions)
        self.curr_policy = np.full(self.num_actions, 1/self.num_actions)
        self.sum_policy = self.curr_policy.copy()
        self.learned_policy = self.curr_policy.copy()
        self.niter = 1

    def regret_matching(self):
        positive = np.maximum(self.cum_regrets, 0.0)
        total = positive.sum()
        if total > 0:
            self.curr_policy = positive / total
        else:
            self.curr_policy = np.full(self.num_actions, 1 / self.num_actions)

    def update(self, utility, node_utility, probability) -> None:
        agent_idx = self.game.agent_name_mapping[self.agent]
        own_reach = probability[agent_idx]
        cf_reach = np.prod(np.delete(probability, agent_idx))

        # accumulate counterfactual regrets
        self.cum_regrets += cf_reach * (utility - node_utility)

        # accumulate average strategy weighted by own reach
        self.sum_policy += own_reach * self.curr_policy
        self.niter += 1

        total = self.sum_policy.sum()
        if total > 0:
            self.learned_policy = self.sum_policy / total
        else:
            self.learned_policy = np.full(self.num_actions, 1 / self.num_actions)

        # regret matching policy
        self.regret_matching()

    def policy(self):
        return self.learned_policy

class CounterFactualRegret(Agent):

    def __init__(self, game: AlternatingGame, agent: AgentID) -> None:
        super().__init__(game, agent)
        self.node_dict: dict[ObsType, Node] = {}

    def action(self):
        try:
            node = self.node_dict[self.game.observe(self.agent)]
            a = np.argmax(np.random.multinomial(1, node.policy(), size=1))
            return a
        except:
            #raise ValueError('Train agent before calling action()')
            print('Node does not exist. Playing random.')
            return np.random.choice(self.game.available_actions())
    
    def train(self, niter=1000):
        for _ in range(niter):
            _ = self.cfr()

    def cfr(self):
        game = self.game.clone()
        utility: dict[AgentID, float] = dict()
        for agent in self.game.agents:
            game.reset()
            probability = np.ones(game.num_agents)
            utility[agent] = self.cfr_rec(game=game, agent=agent, probability=probability)

        return utility 

    def cfr_rec(self, game: AlternatingGame, agent: AgentID, probability: ndarray):
        # TODO
        node_utility = 0 #remove

        return node_utility
        
