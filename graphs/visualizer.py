import matplotlib.pyplot as plt
from celluloid import Camera
import networkx as nx

from environment.game import Hex


class Visualizer:

    def __init__(self, game_config):
        self.game_config = game_config["hex"]
        self.game_logs = []  # Stores all game-plays over all episodes

        self.G = None
        self.game = None
        self.positions = None
        self.fig = None
        self.camera = None

    def add_game_log(self, game_log: list) -> None:
        """
        Saves a new game_log to the overall log
        :param game_log: a list of game states done in a particular episode
        :return: None
        """
        self.game_logs.append(game_log)

    def animate_latest_game(self):
        # Get the latest game log
        game_index = len(self.game_logs) - 1
        actions = self.game_logs[game_index]  # List of Hex actions
        # Initialize everything
        self.game = Hex(self.game_config)
        self.fig = plt.figure()
        self.camera = Camera(self.fig)
        self.G = self.build_graph()
        self.positions = self.calculate_positions()

        # Draw, action, draw
        for action in actions:
            self.game.perform_action(action)
            self.draw()

        # Animate drawings
        animation = self.camera.animate(repeat=False, interval=500)
        animation.save("./graphs/animations/episode{}_animated.gif".format(game_index + 1), writer='pillow')
        plt.close()

    def build_graph(self):
        # Build the Graph
        G = nx.Graph()
        for cell in self.game.get_cells():
            G.add_node(cell)
            # Add all edges from cell to its neighbours
            neighbours = [(cell, neighbour["cell"]) for neighbour in cell.get_neighbours()]
            G.add_edges_from(neighbours)
        return G

    def draw(self):
        """
        Draw the current state of the board
        """
        self.draw_occupied_cells()
        self.draw_open_cells()
        self.draw_edges()
        plt.title('Hex')
        self.camera.snap()

    def draw_occupied_cells(self):
        """
        Update which cells on the board that are pegs
        """
        reds = [cell for cell in self.game.get_cells() if cell.player == 1]
        blacks = [cell for cell in self.game.get_cells() if cell.player == 2]
        nx.draw_networkx_nodes(self.G, pos=self.positions, nodelist=reds,
                               edgecolors='black', node_color='red', linewidths=2)
        nx.draw_networkx_nodes(self.G, pos=self.positions, nodelist=blacks,
                               edgecolors='black', node_color='black', linewidths=2)

    def draw_open_cells(self):
        """
        Update which cells on the board that are empty
        """
        empty_cells = [cell for cell in self.game.get_cells() if cell.player == 0]
        nx.draw_networkx_nodes(self.G, pos=self.positions, nodelist=empty_cells,
                               edgecolors='black', node_color='white', linewidths=2)

    def draw_edges(self):
        """
        Draw the edges of the board - showing which cell are neighbour with who
        """
        nx.draw_networkx_edges(self.G, pos=self.positions)

    def calculate_positions(self):
        """
        Calculate positions of each cell in the visualization.
        A dictionary with nodes as keys and positions as values. Positions should be sequences of length 2.
        """
        return {cell: (cell.column, -cell.row) for cell in self.game.get_cells()}
