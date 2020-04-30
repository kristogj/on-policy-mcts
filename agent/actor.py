import logging
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
from torch.optim.adagrad import Adagrad
from torch.optim.rmsprop import RMSprop
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
from torch.distributions.categorical import Categorical

from agent.models import ANET


def get_optimizer(model: nn.Module, optim: str, lr: float) -> Optimizer:
    """
    Return the optimizer that corresponds to string optim. Add the parameters from model and set learning rate to lr
    :param model: model to get the parameters from
    :param optim: name of the optimizer
    :param lr: learning rate to use in the optimizer
    :return:
    """
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

    def __init__(self, config: dict, load_actor: bool = False):
        if load_actor:
            self.name = None
            self.anet = None
            self.optimizer = None
        else:
            self.anet = ANET(config)
            self.optimizer = self.optimizer = get_optimizer(self.anet.model, config["actor_optim"], config["lr_actor"])

        self.losses = []

    def load_anet(self, name: str, anet: ANET):
        """
        Load pre-trained ANET into the Actor
        :param name: filename of the pre-trained network
        :param anet: pre-trained model
        :return: None
        """
        self.name = name.split("/")[-1]
        self.anet = anet

    def get_conditional_distribution(self, player: int, state: any) -> torch.Tensor:
        """
        Forward the game_state through ANET and return the conditional softmax of the output
        :param player: players turn
        :param state: state of the game being played
        :return: a probability distribution of all legal actions
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

    @staticmethod
    def random_policy(state: any) -> int:
        """
        Return a random action in the distribution
        :param state: current state of the game
        :return: index of the action in the distribution
        """
        legal_indexes = []
        for i, p in enumerate(state):
            if p == 0:
                legal_indexes.append(i)
        return random.choice(legal_indexes)

    def default_policy(self, player: int, state: any, epsilon: float) -> int:
        """
        Forward the state through the network, and return the index of action in distribution
        :param player: players turn
        :param state: state of the game being played
        :param epsilon: probability of doing a random move
        :return: index of action selected from the distribution D
        """
        if random.random() < epsilon:
            action_index = self.random_policy(state)
        else:
            D = self.get_conditional_distribution(player, state)
            action_index = torch.argmax(D).item()
        return action_index

    def topp_policy(self, player: int, state: any) -> int:
        """
        Return the index of action in distribution
        :param player: players turn
        :param state: state of the game being played
        :return: index of action selected from the distribution D
        """
        D = self.get_conditional_distribution(player, state)
        action_index = Categorical(D).sample().item()
        return action_index

    def train(self, batch: list) -> None:
        """
        Trained ANET in a supervised way where you pass the game_state (player+state) through the network, and use the
        distribution D as target
        :param batch: a list of (node, D) tuples from selected from the ReplayBuffer
        :return: None
        """
        X = torch.as_tensor([[node.player] + node.state for node, _ in batch], dtype=torch.float)
        target = torch.as_tensor([torch.argmax(D).item() for _, D in batch], dtype=torch.long)
        # TODO: Check target here
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

    def visualize_loss(self) -> None:
        """
        Visualize the losses saved during training in a plot over episodes
        :return: None
        """
        episodes = list(range(1, len(self.losses) + 1))
        plt.title("Loss over episodes for ANET")
        plt.xlabel("Episodes")
        plt.ylabel("Cross Entropy Loss")
        plt.plot(episodes, self.losses)
        plt.savefig("./graphs/loss.png")
        plt.show()
