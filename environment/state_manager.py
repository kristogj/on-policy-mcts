from utils import get_new_game
from tree_node import Node


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

    def get_child_nodes(self, state):
        """
        Given the state, return all child nodes possible
        :param state:
        :return:
        """
        legal_actions = self.game.get_legal_actions(state)
        new_states = [self.game.get_next_state(state, action) for action in legal_actions]
        return [Node(state, action) for state, action in zip(new_states, legal_actions)]

    def is_winning_state(self, state):
        """
        Check if given state is a winning state
        :param state: Type depends on the game
        :return: boolean
        """
        return self.game.verify_winning_state(state)
