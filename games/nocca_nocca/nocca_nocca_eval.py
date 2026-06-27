import numpy as np
from base.game import AgentID
from games.nocca_nocca.nocca_nocca import NoccaNocca
from games.nocca_nocca.board import BLACK, WHITE, BLACK_GOAL, WHITE_GOAL

class NoccaNoccaEval(NoccaNocca):
    def __init__(self, initial_player=None, max_steps=None, seed=None, render_mode='human', alpha=-1):
        super().__init__(initial_player=initial_player, max_steps=max_steps, seed=seed, render_mode=render_mode)
        self.alpha = alpha

    def eval(self, agent: AgentID) -> float:
        if agent not in self.agents:
            raise ValueError(f"Agent {agent} is not part of the game.")
        
        if self.terminated():
            return self.rewards[agent] * 100
        
        player = self.agent_name_mapping[agent]
        opponent = WHITE if player == BLACK else BLACK
        
        distance = self.distance_to_terminal(player) - self.distance_to_terminal(opponent)
        return self.alpha * distance

    def clone(self):
        self_clone = NoccaNoccaEval(
            initial_player=self.initial_player,
            max_steps=self.max_steps,
            seed=self.seed,
            render_mode=self.render_mode,
            alpha=self.alpha,
        )
        if self.board is not None:
            self_clone.board = self.board.__class__()
            self_clone.board.set_board(self.board)
        self_clone.rewards = self.rewards.copy()
        self_clone.terminations = self.terminations.copy()
        self_clone.truncations = self.truncations.copy()
        self_clone.infos = self.infos.copy()
        self_clone.agent_selection = self.agent_selection
        self_clone.steps = self.steps
        return self_clone

    def distance_to_terminal(self, player) -> int:
        # For Nocca Nocca, the distance is the number of squares between the end of the board and the players nearest free piece
        if player in self.agent_name_mapping:
            player = self.agent_name_mapping[player]

        if player not in [BLACK, WHITE]:
            raise ValueError(f"Invalid player {player}.")

        opponent = WHITE if player == BLACK else BLACK
        goal = BLACK_GOAL if player == BLACK else WHITE_GOAL
        best_distance = float('inf')

        # Squares where player has pieces
        player_squares = np.argwhere(self.board.squares == player)
        for x, y, z in player_squares:
            # Ignore pieces that are blocked by opponent pieces above them
            stack = self.board.squares[x][y]
            if any(piece == opponent for piece in stack[z + 1:]):
                continue

            best_distance = min(best_distance, abs(goal - x))

        if best_distance == float('inf'):
            raise ValueError(f"No unblocked pieces found for player {player}.")

        return int(best_distance)