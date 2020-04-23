import logging
import glob
from models import ANET
import torch


class TournamentOfProgressivePolicies:

    def __init__(self, config, anet_config):
        self.num_games = config["G"]
        self.load_path = config["agent_path"]
        self.anet_config = anet_config
        self.agents = self.load_agents()

    def load_agents(self):
        """
        Load all pre-trained actors networks into a list of agents
        :return: list[ANET]
        """
        file_paths = glob.glob(self.load_path + "/*.pth")
        agents = []
        for file_path in file_paths:
            model = ANET(self.anet_config)
            model.load_state_dict(torch.load(file_path))
            model.eval()
            agents.append((file_path, model))
        return agents
