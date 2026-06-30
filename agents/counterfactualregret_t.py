import numpy as np
from numpy import ndarray
from base.game import AlternatingGame, AgentID, ObsType
from base.agent import Agent
from tqdm import tqdm

class Node():

    def __init__(self, game: AlternatingGame, obs: ObsType) -> None:
        self.game = game
        self.agent = game.agent_selection
        self.obs = obs
        self.num_actions = self.game.num_actions(self.agent)
        self.cum_regrets = np.zeros(self.num_actions)
        self.sum_policy = np.zeros(self.num_actions)

        self.legal_actions = self.game.available_actions()
        self.curr_policy = np.zeros(self.num_actions)
        self.curr_policy[self.legal_actions] = 1 / len(self.legal_actions)

    def regret_matching(self):
        positive = np.maximum(self.cum_regrets, 0.0)
        if positive.sum() > 0:
            self.curr_policy = positive / positive.sum()
        else:
            self.curr_policy = np.zeros(self.num_actions)
            self.curr_policy[self.legal_actions] = 1 / len(self.legal_actions)

    def update(self, utility, node_utility, probability) -> None:
        # Probabilidad de alcance de este nodo (excluyendo al agente actual)
        agent_idx = self.game.agent_name_mapping[self.agent]
        reach = np.prod(np.delete(probability, agent_idx))

        # Acumulo regrets ponderados por la probabilidad de alcance del nodo
        self.cum_regrets[self.legal_actions] += reach * (utility[self.legal_actions] - node_utility)

        # Regret matching para actualizar la política actual
        self.regret_matching()  

        # Acumulo la política actual ponderada por la probabilidad de alcance del nodo
        # CFR llega a equilibrio de Nash en promedio, no en la política actual
        self.sum_policy += probability[agent_idx] * self.curr_policy

    def policy(self):
        total = self.sum_policy.sum()
        if total > 0:
            return self.sum_policy / total
        return self.curr_policy

class CounterFactualRegret(Agent):

    def __init__(self, game: AlternatingGame, agent: AgentID) -> None:
        super().__init__(game, agent)
        self.node_dict: dict[ObsType, Node] = {}

    def action(self):
        obs = self.game.observe(self.agent)
        try:
            node = self.node_dict[obs]
            policy = node.policy()
            a = np.argmax(np.random.multinomial(1, policy, size=1))
            return a
        except KeyError:
            print(f'Node {obs} does not exist. Playing random.')
            return np.random.choice(self.game.available_actions())
    
    def train(self, niter=1000):
        for _ in tqdm(range(niter)):
            _ = self.cfr()

    def cfr(self):
        utility: dict[AgentID, float] = dict()
        for agent in self.game.agents:
            self.game.reset()
            game = self.game.clone()
            probability = np.ones(game.num_agents)
            utility[agent] = self.cfr_rec(game=game, agent=agent, probability=probability)

        return utility 

    def cfr_rec(self, game: AlternatingGame, agent: AgentID, probability: ndarray):
        if game.done():
            return game.reward(agent)

        node_agent = game.agent_selection
        node_agent_idx = game.agent_name_mapping[node_agent]
        obs = game.observe(node_agent)

        if obs not in self.node_dict:
            self.node_dict[obs] = Node(game, obs)
        node = self.node_dict[obs]

        utility = np.zeros(node.num_actions)
        for a in game.available_actions():
            next_game = game.clone()
            next_game.step(a)
            next_probability = probability.copy()
            next_probability[node_agent_idx] = node.curr_policy[a] * probability[node_agent_idx]
            utility[a] = self.cfr_rec(game=next_game, agent=agent, probability=next_probability)

        node_utility = np.dot(node.curr_policy, utility)

        if node_agent == agent:
            node.update(utility, node_utility, probability)

        return node_utility
