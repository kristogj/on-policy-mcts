from utils import get_new_game
from tree_node import Node
import torch
import math
from action import HexAction


def index_to_coordinate(index, size):
    """
    Convert an index from a flat array to a coordinate in a size x size matrix
    :param index: int
    :param size: int
    :return:
    """
    row = math.floor(index / size)
    col = index % size
    return row, col


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

    def get_action(self, player, action_index):
        """
        Convert the action index selected from the output of the Actor Network.
        :param player: int
        :param action_index: int
        :return: Action
        """
        game_type = self.game_config["game_type"]
        if game_type == "hex":
            size = self.game_config["hex"]["board_size"]
            action_coord = index_to_coordinate(action_index, size)
            action = HexAction(player, action_coord)
        else:
            raise ValueError("Action Index to Action object is not supported for this game.")
        return action

    def get_next_state(self, player, current_state, action_index):
        """
        Get the next state of the game by using the action index selected from the output of the Actor Network.
        :param player: int - players turn
        :param current_state: current state of the game being played
        :param action_index: int - index of selected action from a Distribution D
        :return: state of the game after action is performed
        """
        action = self.get_action(player, action_index)
        new_state = self.game.get_next_state(current_state, action)
        return new_state

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
