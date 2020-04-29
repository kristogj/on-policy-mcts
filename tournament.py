import logging
import glob
from models import ANET
import torch
from state_manager import StateManager
import random
from actor import Actor
from utils import get_next_player
import re


class TournamentOfProgressivePolicies:

    def __init__(self, config, anet_config, game_config):
        self.num_games = config["G"]
        self.load_path = config["agent_path"]
        self.anet_config = anet_config
        self.state_manager = StateManager(game_config)
        self.agents = self.load_agents()

    def load_agents(self):
        """
        Load all pre-trained actors networks into a list of agents
        :return: list[ANET]
        """
        logging.info("Loading models for Tournament...")
        file_paths = glob.glob(self.load_path + "/*.pth")
        file_paths.sort(key=lambda name: int(re.findall(r'\d+', name)[0]))
        agents = []
        for file_path in file_paths:
            actor = Actor(None, load_actor=True)
            model = ANET(self.anet_config)
            model.load_state_dict(torch.load(file_path))
            model.eval()
            actor.load_anet(file_path, model)
            agents.append(actor)
        return agents

    def play_game(self, p1, p2):
        """
        Play one game and return the winner
        :param p1: Actor - player 1
        :param p2: Actor - player 2
        :return: int - winner
        """
        actors = {1: p1, 2: p2}
        self.state_manager.init_new_game()
        player = random.randint(1, 2)  # Choose random player to start
        while not self.state_manager.is_winning_state():
            current_state = self.state_manager.get_current_state()
            action_index = actors[player].topp_policy(player, current_state)
            action = self.state_manager.get_action(player, action_index)
            self.state_manager.perform_actual_action(action)
            player = get_next_player(player)

        winner = get_next_player(player)
        return winner

    def start(self):
        """
        Start the tournament, and log results after each series of self.num_games is done
        :return: None
        """
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                p1, p2 = self.agents[i], self.agents[j]
                logging.info("Starting series between {} and {}".format(p1.name, p2.name))
                wins = 0
                for _ in range(self.num_games):
                    winner = self.play_game(p1, p2)
                    wins += int(winner == 2)
                logging.info("{} wins {} of {} games against {} \n".format(p2.name, wins, self.num_games, p1.name))
