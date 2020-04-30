import logging
import random
import torch

from utils import get_next_player
from agent.actor import Actor
from simulator.tree_node import Node
from simulator.mcts import MonteCarloSearchTree
from simulator.replay_buffer import ReplayBuffer
from environment.state_manager import StateManager
from graphs.visualizer import Visualizer


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
        self.epsilon = self.mcts_config["epsilon"]
        self.dr_epsilon = self.mcts_config["dr_epsilon"]
        self.visualize = self.mcts_config["visualize"]
        self.visualize_interval = self.mcts_config["visualize_interval"]

        self.save_interval = self.topp_config["M"]
        self.batch_size = self.anet_config["batch_size"]

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
        save_interval = int(self.episodes / (self.save_interval - 1))  # Save interval for ANET
        rbuf = ReplayBuffer()  # Buffer for saving training data
        visualizer = Visualizer(self.game_config)  # Visualizer that visualize games
        actor = Actor(self.anet_config)  # Initialize Actor which have ANET
        game = StateManager(self.game_config)  # Init a StateManager that takes care of the actual game
        wins = 0  # Number of times player 1 wins

        # Actual games being played
        for episode in range(1, self.episodes + 1):
            logging.info("Episode: {}".format(episode))

            # Initialize the actual game
            game.init_new_game()
            action_log = []

            # Initialize the MonteCarloSearchTree to a single node with the initialized game state
            state, player = game.get_current_state(), self.get_start_player()
            mcts = MonteCarloSearchTree(actor, self.game_config, c=self.mcts_config["c"])
            mcts.set_root(Node(state, None, player=player))

            # While the actual game is not finished
            while not game.is_winning_state():
                # Every time we shall select a new action, we perform M number of simulations in MCTS
                for _ in range(self.num_sim):
                    # One iteration of Monte Carlo Tree Search consists of four steps
                    leaf = mcts.selection()
                    sim_node = mcts.expansion(leaf)
                    z = mcts.simulation(sim_node, epsilon=self.epsilon)
                    mcts.backward(sim_node, z)

                # Get the probability distribution over actions from current root/state.
                D = mcts.get_root_distribution()

                # Add (root, D) to Replay Buffer. This will later be used as training data for the actors policy
                rbuf.add_case((mcts.root, D))

                # Select actual move based on D
                new_root = mcts.select_actual_action(D, player)
                action_log.append(new_root.action)

                # Perform this action, moving the game from state s -> s´
                game.perform_actual_action(new_root.action)

                # Update player
                player = get_next_player(player)

                # Set new root of MCST
                mcts.set_root(new_root)

                # Update epsilon for next round of simulations
                self.epsilon *= self.dr_epsilon

            # End of episode
            visualizer.add_game_log(action_log)

            # Train ANET on a random mini-batch of cases from ReplayBuffer
            mini_batch = rbuf.get_batch(self.batch_size)
            actor.train(mini_batch)

            # Save ANET
            if episode % save_interval == 0 or episode == 1:
                path = "./pretrained/ANET_E{}.pth".format(episode)
                logging.info("Saving model to file {}".format(path))
                torch.save(actor.anet.state_dict(), path)

            # Save visualization of last game
            if self.visualize and episode % self.visualize_interval == 0:
                visualizer.animate_latest_game()

            # If next player is 2 and we are in a win state, player 1 got us in a win state
            if player == 2:
                wins += 1

        actor.visualize_loss()
        logging.info(
            "Player1 wins {} of {} games ({}%)".format(wins, self.episodes, round(100 * (wins / self.episodes))))
