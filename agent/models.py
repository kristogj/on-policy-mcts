import torch.nn as nn
import logging


def get_function(func):
    """
    Return the activation function specified
    :param func: str
    :return:
    """
    if func == "relu":
        return nn.ReLU(inplace=True)
    elif func == "sigmoid":
        return nn.Sigmoid()
    elif func == "tanh":
        return nn.Tanh()
    else:
        raise ValueError("Invalid activation function")


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


class ANET(nn.Module):

    def __init__(self, config):
        super(ANET, self).__init__()
        logging.info("Initializing ANET - {}".format(config))
        self.layer_specs = config["actor_layer_specs"]
        self.layer_functions = config["actor_layer_func"]

        if len(self.layer_specs) - 1 != len(self.layer_functions):
            raise AttributeError("Illegal specs for ANET")

        # Use the given layer specs to initialize the model
        self.model = self.init_model()

    def init_model(self):
        """
        Take the layer specs given, and initialize a Sequential object that represent the model
        :return: nn.Sequential
        """
        model = nn.Sequential()
        for x in range(1, len(self.layer_specs)):
            layer = nn.Linear(in_features=self.layer_specs[x - 1], out_features=self.layer_specs[x])
            model.add_module("L{}".format(x), layer)
            if self.layer_functions[x - 1] != "linear":
                function = get_function(self.layer_functions[x - 1])
                model.add_module("A{}".format(x), function)

        # Initialize weights and biases in the network
        model.apply(init_weights)
        return model

    def forward(self, game_state):
        """
        Take the given state and forward it through the network. Return the output of the network
        :param game_state:
        :return:
        """
        return self.model(game_state)
