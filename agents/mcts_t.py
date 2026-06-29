import sys
import random
from base.game import AlternatingGame, AgentID, ActionType
from base.agent import Agent
from math import log, sqrt
import numpy as np
from typing import Callable

class MCTSNode:
    def __init__(self, parent: 'MCTSNode', game: AlternatingGame, action: ActionType):
        self.parent = parent
        self.game = game
        self.action = action
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.value = 0
        self.cum_rewards = np.zeros(len(game.agents))
        self.agent = self.game.agent_selection
        # acciones aún no expandidas desde este nodo
        self.untried_actions = self.game.available_actions()

def ucb(node, C=sqrt(2)) -> float:
    parent_agent = node.parent.agent 
    agent_idx = node.game.agent_name_mapping[parent_agent]
    return node.cum_rewards[agent_idx] / node.visits + C * sqrt(log(node.parent.visits)/node.visits)

def uct(node: MCTSNode, agent: AgentID) -> MCTSNode:
    child = max(node.children, key=ucb)
    return child

class MonteCarloTreeSearch(Agent):
    def __init__(self, game: AlternatingGame, agent: AgentID, simulations: int=100, rollouts: int=10, rollout_iterations: int=sys.maxsize, selection: Callable[[MCTSNode, AgentID], MCTSNode]=uct) -> None:
        """
        Parameters:
            game: alternating game associated with the agent
            agent: agent id of the agent in the game
            simulations: number of MCTS simulations (default: 100)
            rollouts: number of MC rollouts (default: 10)
            rollout_iterations: number of iterations per rollout (default: 100)
            selection: tree search policy (default: uct)
        """
        super().__init__(game=game, agent=agent)
        self.simulations = simulations
        self.rollouts = rollouts
        self.rollout_iterations = rollout_iterations
        self.selection = selection
        
    def action(self) -> ActionType:
        a, _ = self.mcts()
        return a

    def mcts(self) -> (ActionType, float):

        root = MCTSNode(parent=None, game=self.game.clone(), action=None)

        for i in range(self.simulations):

            node = root
            # node.game = self.game.clone()

            #print(i)
            #node.game.render()

            # selection
            #print('selection')
            node = self.select_node(node=node)

            # expansion
            #print('expansion')
            node = self.expand_node(node)

            # rollout
            #print('rollout')
            rewards = self.rollout(node)

            #update values / Backprop
            #print('backprop')
            self.backprop(node, rewards)

        #print('root childs')
        #for child in root.children:
        #    print(child.action, child.cum_rewards / child.visits)

        action, value = self.action_selection(root)

        return action, value

    def backprop(self, node, rewards):
        while node is not None:
            node.visits += 1
            node.cum_rewards += rewards
            node = node.parent

    def rollout(self, node: MCTSNode):
        rewards = np.zeros(len(self.game.agents))

        for _ in range(self.rollouts):
            game = node.game.clone()
            iterations = self.rollout_iterations
            while not game.game_over() and iterations > 0:
                action = random.choice(game.available_actions())
                game.step(action)
                iterations -= 1

            if game.eval is not None:
                rewards += np.array([game.eval(agent) for agent in game.agents])
            else:
                rewards += np.array([game.reward(agent) for agent in game.agents])
            
        rewards /= self.rollouts
        return rewards

    def select_node(self, node: MCTSNode) -> MCTSNode:
        curr_node = node
        while curr_node.children:
            if curr_node.untried_actions:
                return curr_node
            else:
                curr_node = self.selection(curr_node, self.agent)
        return curr_node

    def expand_node(self, node: MCTSNode) -> None:
        if node.game.game_over():
            return node
        
        action = node.untried_actions.pop(random.randrange(len(node.untried_actions)))
        child_game = node.game.clone()
        child_game.step(action)
        child_node = MCTSNode(parent=node, game=child_game, action=action)
        node.children.append(child_node)
        return child_node

    def action_selection(self, node: MCTSNode) -> (ActionType, float):
        action: ActionType = None
        value: float = float('-inf')

        agent_index = node.game.agent_name_mapping[self.agent]

        for child in node.children:
            child_value = child.cum_rewards[agent_index] / child.visits
            if child_value > value:
                value = child_value
                action = child.action

        return action, value