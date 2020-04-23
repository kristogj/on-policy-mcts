import logging
import yaml


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
