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
        self.curr_policy = np.full(self.num_actions, 1/self.num_actions)
        self.sum_policy = self.curr_policy.copy()
        self.learned_policy = self.curr_policy.copy()
        self.niter = 1

    def masked_policy(self, policy, legal_actions):
        masked = np.zeros(self.num_actions)
        masked[legal_actions] = policy[legal_actions]
        total = masked.sum()
        if total > 0:
            return masked / total

        masked[legal_actions] = 1 / len(legal_actions)
        return masked

    def regret_matching(self, legal_actions):
        positive = np.maximum(self.cum_regrets, 0.0)
        self.curr_policy = self.masked_policy(positive, legal_actions)
        
        self.niter += 1
    
    def update(self, utility, node_utility, probability, legal_actions) -> None:
        # counterfactual reach: product of reach probabilities of all players except this agent
        agent_idx = self.game.agent_name_mapping[self.agent]
        cf_reach = np.prod(np.delete(probability, agent_idx))

        # accumulate counterfactual regrets
        self.cum_regrets[legal_actions] += cf_reach * (utility[legal_actions] - node_utility)

        # regret matching policy (also updates the cumulative average policy)
        self.regret_matching(legal_actions)  

        # own_reach = probability[agent_idx]
        self.sum_policy += self.curr_policy
        self.learned_policy = self.sum_policy / self.niter

    def policy(self, legal_actions=None):
        if legal_actions is not None:
            return self.masked_policy(self.learned_policy, legal_actions)

        return self.learned_policy

class CounterFactualRegret(Agent):

    def __init__(self, game: AlternatingGame, agent: AgentID) -> None:
        super().__init__(game, agent)
        self.node_dict: dict[ObsType, Node] = {}

    def action(self):
        obs = self.game.observe(self.agent)
        try:
            node = self.node_dict[obs]
            policy = node.policy(self.game.available_actions())
            a = np.argmax(np.random.multinomial(1, policy, size=1))
            return a
        except KeyError:
            #raise ValueError('Train agent before calling action()')
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
        # terminal state: return the reward for the agent we are computing the value for
        if game.done():
            return game.reward(agent)

        # player to move at this state
        node_agent = game.agent_selection
        node_agent_idx = game.agent_name_mapping[node_agent]
        obs = game.observe(node_agent)

        # retrieve (or create) the information set node
        if obs not in self.node_dict:
            self.node_dict[obs] = Node(game, obs)
        node = self.node_dict[obs]
        legal_actions = game.available_actions()
        curr_policy = node.masked_policy(node.curr_policy, legal_actions)

        # evaluate the utility of every available action
        utility = np.zeros(node.num_actions)
        for a in legal_actions:
            next_game = game.clone()
            next_game.step(a)
            next_probability = probability.copy()
            next_probability[node_agent_idx] = curr_policy[a] * probability[node_agent_idx]
            utility[a] = self.cfr_rec(game=next_game, agent=agent, probability=next_probability)

        # expected utility of this node under the current policy
        node_utility = np.dot(curr_policy, utility)

        # only update regrets when it is this agent's decision node
        if node_agent == agent:
            node.update(utility, node_utility, probability, legal_actions)

        return node_utility
        
