import sys
import random
from math import log, sqrt
from typing import Callable

import numpy as np

from base.game import AlternatingGame, AgentID, ActionType
from base.agent import Agent


class ISMCTSNode:
    def __init__(self, parent: 'ISMCTSNode', game: AlternatingGame, action: ActionType):
        self.parent = parent
        self.game = game
        self.action = action
        self.children = []
        self.visits = 0
        self.cum_rewards = np.zeros(len(game.agents))
        self.agent = self.game.agent_selection
        self.obs = self.game.observe(self.agent)
        self.untried_actions = [] if self.game.game_over() else self.game.available_actions()

def action_children(node: ISMCTSNode, action: ActionType):
    return [child for child in node.children if child.action == action]

def tried_actions(node: ISMCTSNode):
    return set(child.action for child in node.children)

def find_child(node: ISMCTSNode, action: ActionType, game: AlternatingGame):
    agent = game.agent_selection
    obs = game.observe(agent)

    for child in action_children(node, action):
        if child.agent == agent and child.obs == obs:
            return child

    return None


def ucb(node: ISMCTSNode, action: ActionType, C=sqrt(2)) -> float:
    children = action_children(node, action)
    visits = sum(child.visits for child in children)

    if visits == 0:
        return float('inf')

    agent_idx = node.game.agent_name_mapping[node.agent]
    rewards = sum(child.cum_rewards[agent_idx] for child in children)
    return rewards / visits + C * sqrt(log(node.visits) / visits)

def uct(node: ISMCTSNode, game: AlternatingGame) -> ActionType:
    return max(game.available_actions(), key=lambda action: ucb(node, action))

class ISMCTS(Agent):
    def __init__(self, game: AlternatingGame, agent: AgentID, simulations: int=100, rollouts: int=10, rollout_iterations: int=sys.maxsize, selection: Callable[[ISMCTSNode, AlternatingGame], ActionType]=uct, determinization: Callable[[AlternatingGame, AgentID], AlternatingGame]=None) -> None:
        """
        Parameters:
            game: alternating game associated with the agent
            agent: agent id of the agent in the game
            simulations: number of ISMCTS simulations (default: 100)
            rollouts: number of MC rollouts (default: 10)
            rollout_iterations: number of iterations per rollout (default: sys.maxsize)
            selection: tree search policy (default: uct)
            determinization: samples a game state consistent with this agent's information set
        """
        super().__init__(game=game, agent=agent)
        self.simulations = simulations
        self.rollouts = rollouts
        self.rollout_iterations = rollout_iterations
        self.selection = selection
        self.determinization = determinization

    def action(self) -> ActionType:
        a, _ = self.ismcts()
        return a

    def ismcts(self) -> tuple[ActionType, float]:
        root = ISMCTSNode(parent=None, game=self.determinize(self.game), action=None)

        for _ in range(self.simulations):
            game = self.determinize(self.game)

            node, expanded = self.select_node(node=root, game=game)
            if not expanded:
                node = self.expand_node(node=node, game=game)

            rewards = self.rollout(game)
            self.backprop(node, rewards)

        action, value = self.action_selection(root)

        return action, value

    def determinize(self, game: AlternatingGame) -> AlternatingGame:
        if self.determinization is not None:
            return self.determinization(game, self.agent)

        if hasattr(game, 'random_change'):
            return game.random_change(self.agent)

        return game.clone()

    def select_node(self, node: ISMCTSNode, game: AlternatingGame) -> tuple[ISMCTSNode, bool]:
        curr_node = node

        while not game.game_over():
            if curr_node.untried_actions:
                return curr_node, False

            action = self.selection(curr_node, game)
            game.step(action)

            child = find_child(curr_node, action, game)
            if child is None:
                child = ISMCTSNode(parent=curr_node, game=game.clone(), action=action)
                curr_node.children.append(child)
                return child, True

            curr_node = child

        return curr_node, True

    def expand_node(self, node: ISMCTSNode, game: AlternatingGame) -> ISMCTSNode:
        if game.game_over():
            return node

        action = node.untried_actions.pop(random.randrange(len(node.untried_actions)))
        game.step(action)
        child_node = ISMCTSNode(parent=node, game=game.clone(), action=action)
        node.children.append(child_node)
        return child_node

    def rollout(self, game: AlternatingGame):
        rewards = np.zeros(len(self.game.agents))

        for _ in range(self.rollouts):
            rollout_game = game.clone()
            iterations = self.rollout_iterations

            while not rollout_game.game_over() and iterations > 0:
                action = random.choice(rollout_game.available_actions())
                rollout_game.step(action)
                iterations -= 1

            if getattr(rollout_game, 'eval', None) is not None:
                rewards += np.array([rollout_game.eval(agent) for agent in rollout_game.agents])
            else:
                rewards += np.array([rollout_game.reward(agent) for agent in rollout_game.agents])

        rewards /= self.rollouts
        return rewards

    def backprop(self, node, rewards):
        while node is not None:
            node.visits += 1
            node.cum_rewards += rewards
            node = node.parent

    def action_selection(self, node: ISMCTSNode) -> tuple[ActionType, float]:
        action: ActionType = None
        value: float = float('-inf')

        agent_index = node.game.agent_name_mapping[self.agent]

        for child_action in tried_actions(node):
            children = action_children(node, child_action)
            visits = sum(child.visits for child in children)

            if visits == 0:
                continue

            child_value = sum(child.cum_rewards[agent_index] for child in children) / visits
            if child_value > value:
                value = child_value
                action = child_action

        return action, value
