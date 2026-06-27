from base.agent import Agent, AgentID
from base.game import AlternatingGame
import numpy as np
import sys

class MiniMaxAlphaBeta(Agent):

    def __init__(self, game: AlternatingGame, agent: AgentID, seed=None, depth: int=sys.maxsize, alphabeta: bool=False) -> None:
        super().__init__(game, agent)

        if depth < 0:
            raise ValueError("Depth must be a non-negative integer.")

        self.depth = depth
        self.alphabeta = alphabeta
        self.iterations = 0
        
        self.seed = seed
        np.random.seed(seed)
    
    def action(self):
        self.iterations = 0
        act, _ = self.minimax(self.game, self.depth)
        return act

    def minimax(self, game: AlternatingGame, depth: int, alpha: float=float('-inf'), beta: float=float('inf')):
        self.iterations += 1

        agent = game.agent_selection
        chosen_action = None  

        #Casos base

        if game.terminated():             
            return None, game.reward(self.agent)

        if depth == 0:
            return None, self.eval(game)
        
        #Casos no base

        actions = game.available_actions()
        np.random.shuffle(actions)

        if agent != self.agent: # Min
            value = float('inf')
            for action in actions:
                child = game.clone()
                child.step(action)

                _, minimax_value = self.minimax(child, depth-1, alpha, beta)
                if minimax_value < value:
                    value = minimax_value
                    chosen_action = action

                if self.alphabeta:
                    beta = min(beta, value)
                    if beta <= alpha:
                        break

        else: # Max (player == self.player)
            value = float('-inf')
            for action in actions:
                child = game.clone()
                child.step(action)

                _, minimax_value = self.minimax(child, depth-1, alpha, beta)
                if minimax_value > value:
                    value = minimax_value
                    chosen_action = action

                if self.alphabeta:
                    alpha = max(alpha, value)
                    if beta <= alpha:
                        break

        return chosen_action, value

    def eval(self, game: AlternatingGame):
        return game.eval(self.agent)