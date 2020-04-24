import logging
from mcts import MonteCarloSearchTree
from tree_node import Node
import random
from utils import get_next_player, get_new_game, save_model
from actor import Actor
import torch
from replay_buffer import ReplayBuffer


class GameSimulator:
    """
    The simulator must provide these results:
        Play-by-Play game action when in verbose mode
        Essential win statistics (for at least one of the two players) for a batch of games.
    """

    def __init__(self, config):
        """
        Config consists of attributes used in the Game Simulator
        :param config: dict
        """
        self.game_config = config["game_config"]
        self.mcts_config = config["mcts_config"]
        self.anet_config = config["anet_config"]
        self.topp_config = config["topp_config"]

        self.starting_player = self.mcts_config["starting_player"]
        self.episodes = self.mcts_config["episodes"]
        self.num_sim = self.mcts_config["num_sim"]

        self.save_interval = self.topp_config["M"]

        logging.info("Initializing GameSimulator - {}".format(self.game_config["game_type"]))

    def get_start_player(self):
        """
        Return the starting player, choosing a random if starting player equals 3.
        :return: int
        """
        player = self.starting_player
        if player == 3:
            player = random.randint(1, 2)
        return player

    def simulate(self):
        """
        Run G consecutive games (aka. episodes) of the self.game_type using fixed values for the game parameters
        """
        # SAve interval for ANET (the actor network) parameters

        save_interval = int(self.episodes / (self.save_interval - 1))

        # TODO: Clear Replay Buffer (RBUF)
        rbuf = ReplayBuffer()

        # Initialize ANET
        actor = Actor(self.anet_config)
        wins = 0  # Number of times player 1 wins

        # Actual games being played
        for episode in range(1, self.episodes + 1):
            logging.info("Episode: {}".format(episode))

            # Initialize the actual game
            game = get_new_game(self.game_config)

            # Initialize the MonteCarloSearchTree to a single node with the initialized game state
            state, player = game.get_current_state(), self.get_start_player()
            mcts = MonteCarloSearchTree(self.game_config, c=self.mcts_config["c"])
            mcts.set_root(Node(state, None, player=player))

            # While the actual game is not finished
            while not game.is_winning_state():
                # Every time we shall select a new action, we perform M number of simulations in MCTS
                for _ in range(self.num_sim):
                    # One iteration of Monte Carlo Tree Search consists of four steps
                    # 1. Selection
                    leaf = mcts.selection()
                    # 2. Expand selected leaf node
                    sim_node = mcts.expansion(leaf)
                    # 3. Simulation
                    z = mcts.simulation(sim_node)
                    # 4. Backward propagation
                    mcts.backward(sim_node, z)

                # TODO: distribution of visit counts in MCT along all arcs emanating from root
                D = None

                # TODO: Add case (root, D) to RBUF
                rbuf.add_case((mcts.root, D))

                # TODO: Now use the search tree to choose next action
                new_root = mcts.select_actual_action(D, player)

                # Perform this action, moving the game from state s -> s´
                game.perform_action(new_root.action)

                # Update player
                player = get_next_player(player)

                # Set new root of MCST
                mcts.set_root(new_root)

            # End of episode
            # Train ANET on a random mini-batch of cases from RBUF

            # Save ANET
            if episode % save_interval == 0 or episode == 1:
                torch.save(actor.anet.state_dict(), "./pretrained/ANET_E{}.pth".format(episode))

            # If next player is 2 and we are in a win state, player 1 got us in a win state
            if player == 2:
                wins += 1

        # Report statistics
        logging.info(
            "Player1 wins {} of {} games ({}%)".format(wins, self.episodes, round(100 * (wins / self.episodes))))
