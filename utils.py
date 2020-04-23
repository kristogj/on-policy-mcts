import logging
import yaml
from game import Hex
import torch


def init_logger():
    """
    Initialize logger settings
    :return: None
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler("app.log", mode="w"),
            logging.StreamHandler()
        ])


def load_config(path):
    """
    Load the configuration from task_2_table.yaml.
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)


def get_next_player(player):
    return 2 if player == 1 else 1


def get_new_game(game_config):
    """
    Initialize a new Hex game, and return it
    :param game_config: dict
    :return: Game
    """
    _type = game_config["game_type"]
    if _type == "hex":
        game = Hex(game_config["hex"], verbose=game_config["verbose"])
    else:
        raise ValueError("Game type is not supported")
    return game


def save_model(model, path):
    torch.save(model.state_dict(), path)
