from utils import get_new_game
from tree_node import Node
import torch


class StateManager:

    def __init__(self, game_config):
        self.game_config = game_config
        self.game = None

    def init_new_game(self):
        """
        Initialize a new game for MCTS, this should not print any updates about the game as it will not be used
        in a chronological game order
        :return: None
        """
        self.game = get_new_game(self.game_config)

    def get_child_nodes(self, player, state):
        """
        Given the state, return all child nodes possible
        :param player: int - Player making an action from that state
        :param state: list[int]
        :return:
        """
        legal_actions = self.game.get_legal_actions(player, state)
        new_states = [self.game.get_next_state(state, action) for action in legal_actions]
        return [Node(state, action) for state, action in zip(new_states, legal_actions)]

    def verify_winning_state(self, state):
        """
        Check if given state is a winning state
        :param state: Type depends on the game
        :return: boolean
        """
        return self.game.verify_winning_state(state)

    def is_winning_state(self):
        """
        Check if the actual game being played is in a winning state
        :return: boolean
        """
        return self.game.is_winning_state()

    def perform_actual_action(self, action):
        """
        Perform the given action in the actual game
        :param action: Action
        :return: None
        """
        self.game.perform_action(action)

    def get_current_state(self):
        """
        Return the current state of the game
        :return:
        """
        return self.game.get_current_state()

    def get_node_distribution(self, root):
        """
        # TODO: Should it be normalized?
        From the node given, calculate the (normalized?)distribution D
        :param root:
        :return:
        """
        game_type = self.game_config["game_type"]
        D = None
        if game_type == "hex":
            # TODO: Is this correct way to calculate D for Hex?
            size = self.game_config["hex"]["board_size"]
            D = torch.zeros((1, size, size))
            for child in root.children:
                row, col = child.action.get_coord()
                D[0][row][col] = child.total
            # Flatten D
            D = D.flatten(1)

            # Normalize values to be smaller
            D = (D - D.mean()) / D.std()

            # TODO: Duplicate code, could be refactored
            # Calculate exp before re-normalizing softmax
            D = torch.exp(D)
            # Set positions that are already taken to zero
            mask = torch.IntTensor([int(player == 0) for player in root.state])
            D[0] *= mask
            # Re-normalize values that are not equal to zero to sum up to 1
            all = torch.sum(D)
            D /= all
            D = D[0]
        return D
