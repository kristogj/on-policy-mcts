import torch
import torch.nn as nn
import logging


def get_function(func: str) -> nn.Module:
    """
    Return the activation function specified
    :param func: name of activation function to return
    :return: the activation function
    """
    if func == "relu":
        return nn.ReLU(inplace=True)
    elif func == "sigmoid":
        return nn.Sigmoid()
    elif func == "tanh":
        return nn.Tanh()
    else:
        raise ValueError("Invalid activation function")


def init_weights(m: nn.Module) -> None:
    """
    Initialize the weights of the module
    :param m: a module with parameters that could be initialized
    :return: None
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def get_model(layer_specs, layer_functions):
    model = nn.Sequential()
    for x in range(1, len(layer_specs)):
        layer = nn.Linear(in_features=layer_specs[x - 1], out_features=layer_specs[x])
        model.add_module("L{}".format(x), layer)
        if layer_functions[x - 1] != "linear":
            function = get_function(layer_functions[x - 1])
            model.add_module("A{}".format(x), function)

    # Initialize weights and biases in the network
    model.apply(init_weights)
    return model


class ANET(nn.Module):

    def __init__(self, config: dict) -> None:
        super(ANET, self).__init__()
        logging.info("Initializing ANET - {}".format(config))
        self.layer_specs = config["actor_layer_specs"]
        self.layer_functions = config["actor_layer_func"]

        if len(self.layer_specs) - 1 != len(self.layer_functions):
            raise AttributeError("Illegal specs for ANET")

        # Use the given layer specs to initialize the model
        self.model = get_model(self.layer_specs, self.layer_functions)

    def forward(self, game_state: torch.Tensor) -> torch.Tensor:
        """
        Take the given state and forward it through the network. Return the output of the network
        :param game_state: input to the model
        :return: output from the model
        """
        # TODO: Could it be smart to turn 2s into -1 to get the values in the range {-1,1}
        game_state.apply_(lambda x: -1 if x == 2 else x)
        return self.model(game_state)


class CNET(nn.Module):

    def __init__(self, config):
        super(CNET, self).__init__()
        logging.info("Initializing CNET - {}".format(config))
        self.layer_specs = config["critic_layer_specs"]
        self.layer_functions = config["critic_layer_func"]

        if len(self.layer_specs) - 1 != len(self.layer_functions):
            raise AttributeError("Illegal specs for CNET")

        # Use the given layer specs to initialize the model
        self.model = get_model(self.layer_specs, self.layer_functions)

    def forward(self, game_state: torch.Tensor) -> torch.Tensor:
        game_state.apply_(lambda x: -1 if x == 2 else x)
        return self.model(game_state)
