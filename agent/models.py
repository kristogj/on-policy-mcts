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


class ANET(nn.Module):

    def __init__(self, config: dict) -> None:
        super(ANET, self).__init__()
        logging.info("Initializing ANET - {}".format(config))
        self.layer_specs = config["layer_specs"]
        self.layer_functions = config["layer_func"]

        if len(self.layer_specs) - 1 != len(self.layer_functions):
            raise AttributeError("Illegal specs for ANET")

        # Use the given layer specs to initialize the model
        self.model = self.get_model()

    def get_model(self):
        model = nn.Sequential()
        for x in range(1, len(self.layer_specs)):
            if x == len(self.layer_specs) - 1:
                layer = DuoOutputLayer(in_features=self.layer_specs[x - 1], out_features1=self.layer_specs[x],
                                       out_features2=1)
            else:
                layer = nn.Linear(in_features=self.layer_specs[x - 1], out_features=self.layer_specs[x])
            model.add_module("L{}".format(x), layer)
            if self.layer_functions[x - 1] != "linear":
                function = get_function(self.layer_functions[x - 1])
                model.add_module("A{}".format(x), function)

        # Initialize weights and biases in the network
        model.apply(init_weights)
        return model

    def forward(self, game_state: torch.Tensor) -> torch.Tensor:
        """
        Take the given state and forward it through the network. Return the output of the network
        :param game_state: input to the model
        :return: output from the model
        """
        game_state.apply_(lambda x: -1 if x == 2 else x)
        return self.model(game_state)


class DuoOutputLayer(nn.Module):

    def __init__(self, in_features, out_features1, out_features2):
        super(DuoOutputLayer, self).__init__()
        self.out1 = nn.Linear(in_features, out_features1)
        self.out2 = nn.Linear(in_features, out_features2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, state: torch.Tensor):
        return self.out1(state), self.sigmoid(self.out2(state))
