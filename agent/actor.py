from models import ANET
from torch.optim.adagrad import Adagrad
from torch.optim.rmsprop import RMSprop
from torch.optim.adam import Adam
from torch.optim.sgd import SGD
import logging


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
        pass
