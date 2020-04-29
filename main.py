from utils import init_logger, load_config
from game_simulator import GameSimulator
from tournament import TournamentOfProgressivePolicies

if __name__ == '__main__':
    init_logger()
    config = load_config("configs/config.yaml")
    game_simulator = GameSimulator(config)
    #game_simulator.simulate()

    # Play TOPP
    topp = TournamentOfProgressivePolicies(config["topp_config"], config["anet_config"], config["game_config"])
    topp.start()
