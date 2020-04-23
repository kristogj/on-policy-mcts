from utils import init_logger, load_config
from game_simulator import GameSimulator

if __name__ == '__main__':
    init_logger()
    config = load_config("configs/config.yaml")
    game_simulator = GameSimulator(config)
    game_simulator.simulate()
