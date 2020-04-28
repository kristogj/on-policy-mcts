from models import ANET
import logging

import torch
import torch.nn.functional as F
from torch.optim.adagrad import Adagrad
from torch.optim.rmsprop import RMSprop
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.distributions.categorical import Categorical


def get_optimizer(model, optim, lr):
    if optim == "adagrad":
        return Adagrad(model.parameters(), lr=lr)
    elif optim == "sgd":
        return SGD(model.parameters(), lr=lr)
    elif optim == "rmsprop":
        return RMSprop(model.parameters(), lr=lr)
    elif optim == "adam":
        return Adam(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer")


class Actor:

    def __init__(self, config):
        self.anet = ANET(config)
        self.optimizer = self.optimizer = get_optimizer(self.anet.model, config["actor_optim"], config["lr_actor"])

    def get_raw_distribution(self, player, state):
        """
        Forward the game_state through ANET and return the softmax of the output
        """
        # First element of the list represent which player turn it is
        game_state = [player] + state
        tensor_state = torch.FloatTensor(game_state).unsqueeze(0)
        # Forward through ANET
        D = self.anet(tensor_state)
        # Apply softmax to output and return un-normalized distribution over actions
        D = F.softmax(D, dim=1)
        return D

    def default_policy(self, player, state):
        """
        Forward the state through the network, and return the new state
        :param player: int
        :param state: list[int]
        :return:
        """
        D = self.get_raw_distribution(player, state)

        # Calculate exp before re-normalizing softmax
        D = torch.exp(D)

        # Set positions that are already taken to zero
        mask = torch.IntTensor([int(player == 0) for player in state])
        D[0] *= mask

        # Re-normalize values that are not equal to zero to sum up to 1
        all = torch.sum(D)
        D /= all

        # In the default policy we just sample
        action_index = Categorical(D).sample()

        new_state = state.copy()
        new_state[action_index.item()] = player

        return new_state
