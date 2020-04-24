from abc import ABC, abstractmethod
from cell import Cell
from collections import deque
from action import HexAction


class Game(ABC):
    """
    Abstract game class
    """

    def __init__(self, verbose):
        self.verbose = verbose

    @abstractmethod
    def perform_action(self, action):
        pass

    @abstractmethod
    def is_winning_state(self):
        pass

    @staticmethod
    @abstractmethod
    def verify_winning_state(state):
        pass

    @abstractmethod
    def get_legal_actions(self, player, state):
        pass

    @abstractmethod
    def get_next_state(self, state, action):
        pass

    @abstractmethod
    def get_current_state(self):
        pass


class Hex(Game):

    def __init__(self, config, verbose=False):
        super(Hex, self).__init__(verbose)
        self.size = config["board_size"]
        self.board = [[None for _ in range(self.size)] for _ in range(self.size)]  # The actual board used while playing

        # Each cell will have a maximum of 6 neighbours: (r-1,c) (r-1,c+1), (r,c-1), (r.c+1), (r+1,c-1), (r+1,c)
        self.neighbour_pattern = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]

        # Initialize board with empty cells, and calculate the neighbours before game start
        self.init_board()

    def init_board(self):
        """
        Update board with new Cell instances
        """
        for row in range(self.size):
            for column in range(self.size):
                self.set_cell(Cell(row, column, 0))

        self.set_neighbours()

    def get_cell(self, coord):
        """
        Return cell at the given coordinate on the board
        :param coord: tuple (row, col)
        :return: Cell
        """
        row, col = coord
        return self.board[row][col]

    def get_cells(self):
        """
        Return the cells on the board as a 1D list
        :return: list[Cell]
        """
        return [cell for row in self.board for cell in row]

    def get_legal_actions(self, player, state):
        """
        All empty cells on the board are legal actions
        :param player: int - Player performing the action
        :param state: list[int] - List of players
        :return:
        """
        if len(state) != self.size ** 2:
            raise ValueError("Illegal state given to get_legal_actions")
        self.update_state(state)
        return [HexAction(player, cell.coord) for cell in self.get_cells() if cell.get_player() == 0]

    def get_current_state(self):
        """
        Return the current state of the board - which in this case is a list of a cells player
        :return: list[int]
        """
        return [cell.player for cell in self.get_cells()]

    def get_next_state(self, state, action):
        self.update_state(state)
        self.perform_action(action)
        return [cell.get_player() for cell in self.get_cells()]

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
        """
        Iterate over all cells on top and left of the board. On top look for player 1 cells, and on the left side
        look for player 2 cells. If you find a cell for the corresponding player, do a depth first search to see
        if there is a connecting path from that cell to the other side of the board.
        :return: boolean
        """
        for x in range(self.size):
            p1_cell, p2_cell = self.board[0][x], self.board[x][0]
            w1, w2 = False, False
            if p1_cell.get_player() == 1:
                w1 = self.dfs(p1_cell, p1_cell.get_player())
            if p2_cell.get_player() == 2:
                w2 = self.dfs(p2_cell, p2_cell.get_player())
            if w1 or w2:
                return True
        return False

    def dfs(self, cell, player):
        """
        Use depth first search to see if there is a path from one side of the board to the other.
        The sides being checked depends on the player.
        Player 1 (red) has top and bottom side of the board
        Player 2 (black) has left and right side of the board.
        :param cell: Cell
        :param player: int
        :return: boolean
        """
        stack = deque()
        discovered = set()
        stack.append(cell)
        while len(stack) != 0:
            current = stack.pop()
            # If you get to a cell from the other side of the board, this means that there must be a path over the board
            if (player == 1 and (current.row == self.size - 1)) or (player == 2 and (current.column == self.size - 1)):
                return True
            if current not in discovered:
                discovered.add(current)
                for neighbour in current.get_neighbours():
                    # If the neighbour also belong to player, we have a connecting path, add it to the stack
                    # and explore that path further
                    if neighbour["cell"].get_player() == player:
                        stack.append(neighbour["cell"])
        return False

    def update_state(self, state):
        """
        Update the state of the board with the players given in state
        :param state: list[int]
        :return: None
        """
        cells = self.get_cells()
        for cell, player in zip(cells, state):
            cell.set_player(player)

    def verify_winning_state(self, state):
        """
        Update the board to given state, and check if it is a winning state
        :param state: list[int] - list of players for each cell on the board
        :return: boolean
        """
        self.update_state(state)
        return self.is_winning_state()

    def perform_action(self, action):
        """
        Set the player at the given cell to player given in action
        :param action: HexAction
        :return: None
        """
        player, (row, col) = action.player, action.coord
        self.board[row][col].set_player(player)

    def __str__(self):
        res = ""
        for row in self.board:
            res += str(row) + "\n"
        return res


if __name__ == '__main__':
    config = dict()
    config["board_size"] = 2
    game = Hex(config)
    print(game.get_legal_actions(2, [0, 0, 0, 0]))
