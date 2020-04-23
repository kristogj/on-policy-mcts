from abc import ABC, abstractmethod
from cell import Cell


class Game(ABC):
    """
    Abstract game class
    """

    def __init__(self, verbose):
        self.verbose = verbose

    @abstractmethod
    def perform_action(self, player, action):
        pass

    @abstractmethod
    def is_winning_state(self):
        pass

    @staticmethod
    @abstractmethod
    def verify_winning_state(state):
        pass

    @abstractmethod
    def get_legal_actions(self, state):
        pass

    @abstractmethod
    def get_next_state(self, state, action):
        pass


class Hex(Game):

    def __init__(self, config, verbose=False):
        super(Hex, self).__init__(verbose)
        self.size = config["board_size"]
        self.board = [[None for _ in range(self.size)] for _ in range(self.size)]  # The actual board used while playing

        # Each cell will have a maximum of 6 neighbours: (r-1,c) (r-1,c+1), (r,c-1), (r.c+1), (r+1,c-1), (r+1,c)
        self.neighbour_pattern = [(-1, 0), (-1, 1), (0, - 1), (0, 1), (1, - 1), (1, 0)]

        # Initialize board with empty cells, and calculate the neighbours before game start
        self.init_board()

    def init_board(self):
        """
        Update board with new Cell instances
        """
        for row in range(self.size):
            for column in range(self.size):
                self.set_cell(Cell(row, column, False))

        self.set_neighbours()

    def get_cell(self, coord):
        """
        Return cell at the given coordinate on the board
        :param coord: tuple (row, col)
        :return: Cell / None
        """
        row, col = coord
        return self.board[row][col]

    def get_legal_actions(self, state):
        pass

    def get_next_state(self, state, action):
        pass

    def set_cell(self, cell):
        """
        Update board at cell position
        :param cell: Cell
        """
        self.board[cell.row][cell.column] = cell

    def set_neighbours(self):
        """
        For each cell on the board, calculate its neighbours positions and add to list of neighbours if it is a legal
        neighbour.
        :return: None
        """
        for r in range(self.size):
            for c in range(self.size):
                current_cell = self.get_cell((r, c))
                if current_cell:
                    neighbours = list(map(lambda tup: (r + tup[0], c + tup[1]), self.neighbour_pattern))
                    for i, coord in enumerate(neighbours):
                        if self.is_legal_neighbour(coord):
                            cell = self.get_cell(coord)
                            if cell:
                                current_cell.add_neighbour(cell, self.neighbour_pattern[i])

    def is_legal_neighbour(self, coord):
        """
        Check if coord gives position to a legal cell on the board
        :param coord:
        :return:
        """
        r, c = coord
        return (0 <= r < self.size) and (0 <= c < self.size)

    def is_winning_state(self):
        pass

    @staticmethod
    def verify_winning_state(state):
        pass

    def perform_action(self, player, action):
        pass

    def __str__(self):
        res = ""
        for row in self.board:
            res += str(row) + "\n"
        return res
