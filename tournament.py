import logging
import glob
import torch
import random
import re

from agent.models import ANET
from agent.actor import Actor
from environment.state_manager import StateManager
from utils import get_next_player
from graphs.visualizer import Visualizer


class TournamentOfProgressivePolicies:

    def __init__(self, config, anet_config, game_config):
        self.num_games = config["G"]
        self.load_path = config["agent_path"]
        self.anet_config = anet_config
        self.state_manager = StateManager(game_config)
        self.visualizer = Visualizer(game_config)  # Visualizer that visualize games
        self.agents = self.load_agents()

    def load_agents(self):
        """
        Load all pre-trained actors networks into a list of agents
        :return: list[ANET]
        """
        logging.info("Loading models for Tournament...")
        file_paths = glob.glob(self.load_path + "/*.pth")
        file_paths.sort(key=lambda name: int(re.findall(r'\d+', name.split("/")[-1])[0]))
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
        action_log = []
        while not self.state_manager.is_winning_state():
            current_state = self.state_manager.get_current_state()
            action_index = actors[player].topp_policy(player, current_state)
            action = self.state_manager.get_action(player, action_index)
            self.state_manager.perform_actual_action(action)
            player = get_next_player(player)
            action_log.append(action)

        winner = get_next_player(player)
        return actors[winner].name, action_log

    def start(self):
        """
        Start the tournament, and log results after each series of self.num_games is done
        :return: None
        """
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                p = [self.agents[j], self.agents[i]]
                logging.info("Starting series between {} and {}".format(p[0].name, p[1].name))
                score = {p[0].name: 0, p[1].name: 0}
                for _ in range(self.num_games):
                    random.shuffle(p)
                    winner, action_log = self.play_game(p[0], p[1])
                    score[winner] += 1
                    self.visualizer.add_game_log(action_log)
                    if random.random() > 0.98:
                        self.visualizer.animate_latest_game()
                p.sort(key=lambda player: -int(re.findall(r'\d+', player.name)[0]))  # Nicer scoreboard
                p0, p1 = p[0].name, p[1].name
                logging.info("{}: {}, {}: {} \n".format(p0, score[p0], p1, score[p1]))
