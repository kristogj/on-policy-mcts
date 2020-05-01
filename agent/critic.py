import logging
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.optim.adagrad import Adagrad
from torch.optim.rmsprop import RMSprop
from torch.optim.adam import Adam
from torch.optim.sgd import SGD

from agent.models import CNET


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


class Critic:

    def __init__(self, config: dict, load_critic: bool = False):
        if load_critic:
            self.name = None
            self.anet = None
            self.optimizer = None
        else:
            self.cnet = CNET(config)
            self.optimizer = get_optimizer(self.anet.model, config["actor_optim"], config["lr_actor"])

        self.criterion = nn.BCELoss()
        self.losses = []

    def load_anet(self, name: str, cnet: CNET):
        """
        Load pre-trained ANET into the Actor
        :param name: filename of the pre-trained network
        :param anet: pre-trained model
        :return: None
        """
        self.name = name.split("/")[-1]
        self.cnet = cnet

    def value_function(self, player: int, state: any) -> float:
        """
        Take a game state (player + state) and return a probability (0-1) of winning in this position
        :param player: players turn
        :param state: current state of the game
        :return: probability of winning in this state
        """
        # First element of the list represent which player turn it is
        tensor_state = torch.as_tensor([player] + state, dtype=torch.float)
        # Forward through ANET
        return self.cnet(tensor_state)

    def train(self, batch: list) -> None:
        """
        Trained ANET in a supervised way where you pass the game_state (player+state) through the network, and use the
        distribution D as target
        :param batch: a list of (node, reinforcement) tuples from selected from the ReplayBuffer
        :return: None
        """
        X = torch.as_tensor([[node.player] + node.state for node, _ in batch], dtype=torch.float)
        targets = torch.stack([reward for _, reward in batch], dim=0)

        # Zero the parameter gradients from in ANET
        self.optimizer.zero_grad()

        # Forward input through model
        out = self.cnet(X)

        # Cross Entropy loss function
        loss = self.criterion(out, targets)

        # Calculate gradients, and update weights
        loss.backward()
        self.optimizer.step()

        # Save for later graphing
        self.losses.append(loss.item())
        logging.info("Critic Loss: {}".format(loss.item()))

    def visualize_loss(self) -> None:
        """
        Visualize the losses saved during training in a plot over episodes
        :return: None
        """
        episodes = list(range(1, len(self.losses) + 1))
        plt.title("Loss over episodes for CNET")
        plt.xlabel("Episodes")
        plt.ylabel("Cross Entropy Loss")
        plt.plot(episodes, self.losses)
        plt.savefig("./graphs/loss_critic.png")
        plt.show()
