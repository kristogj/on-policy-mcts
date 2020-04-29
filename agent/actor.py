from models import ANET
import logging
import matplotlib.pyplot as plt
import math
from action import HexAction

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

    def __init__(self, config, load_actor=False):
        if load_actor:
            self.name = None
            self.anet = None
            self.optimizer = None
        else:
            self.anet = ANET(config)
            self.optimizer = self.optimizer = get_optimizer(self.anet.model, config["actor_optim"], config["lr_actor"])

        self.losses = []

    def load_anet(self, name, anet):
        self.name = name.split("/")[-1]
        self.anet = anet

    def get_conditional_distribution(self, player, state):
        """
        Forward the game_state through ANET and return the conditional softmax of the output
        """
        # First element of the list represent which player turn it is
        tensor_state = torch.as_tensor([player] + state, dtype=torch.float)
        # Forward through ANET
        D = self.anet(tensor_state)
        # Apply softmax to output and return un-normalized distribution over actions
        D = F.softmax(D, dim=0)

        # Calculate exp before re-normalizing softmax
        D = torch.exp(D)

        # Set positions that are already taken to zero - TODO: This is depended on what game is being played
        mask = torch.as_tensor([int(player == 0) for player in state], dtype=torch.int)
        D *= mask

        # Re-normalize values that are not equal to zero to sum up to 1
        all = torch.sum(D)
        D /= all

        return D

    def default_policy(self, player, state):
        """
        Forward the state through the network, and return the new state
        :param player: int
        :param state: list[int]
        :return:
        """
        D = self.get_conditional_distribution(player, state)
        # TODO: Could also depend on a value epsilon that decreases. Instead of sampling could then do random or max
        action_index = Categorical(D).sample()
        return action_index

    def topp_policy(self, player, state):
        """
        Return the index of action in distribution
        :param player: int
        :param state: list[int]
        :return:
        """
        D = self.get_conditional_distribution(player, state)
        action_index = torch.argmax(D)
        return action_index

    def train(self, batch):
        """
        Trained ANET in a supervised way where you pass the game_state (player+state) through the network, and use the
        distribution D as target
        :param batch:
        :return:
        """
        X = torch.as_tensor([[node.player] + node.state for node, _ in batch], dtype=torch.float)
        target = torch.as_tensor([torch.argmax(D).item() for _, D in batch], dtype=torch.long)

        # Zero the parameter gradients from in ANET
        self.optimizer.zero_grad()

        # Forward input through model
        out = self.anet(X)
        out = F.softmax(out, dim=1)

        # Calculate loss
        loss = F.cross_entropy(out, target)

        # Calculate gradients, and update weights
        loss.backward()
        self.optimizer.step()

        # Save for later graphing
        self.losses.append(loss.item())
        logging.info("Loss: {}".format(loss.item()))

    def visualize_loss(self):
        episodes = list(range(1, len(self.losses) + 1))
        plt.title("Loss over episodes for ANET")
        plt.xlabel("Episodes")
        plt.ylabel("Cross Entropy Loss")
        plt.plot(episodes, self.losses)
        plt.savefig("./graphs/loss.png")
        plt.show()
