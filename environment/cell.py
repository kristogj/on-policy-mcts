class Cell:

    def __init__(self, row, column, player):
        self.row, self.column = row, column  # Location on board
        self.neighbours = {}  # Neighbour cells on board
        self.player = player  # Which player owns this cell

    def add_neighbour(self, cell, pattern):
        """
        Add cell to the table of neighbours mapping from (row, col) -> {cell, pattern}
        This might be switched out with simple list in the future
        :param cell: Cell
        :param pattern: List[tuple]
        :return:
        """
        self.neighbours[(cell.row, cell.column)] = {"cell": cell, "pattern": pattern}

    def get_neighbours(self):
        """
        Return a list of all neighbour cells
        :return: List[Cell]
        """
        return self.neighbours.values()

    def get_player(self):
        """
        Return the player that occupies this cell
        :return: int
        """
        return self.player

    def set_player(self, player):
        """
        Set player value of this cell to player
        :param player: int
        :return: None
        """
        self.player = player

    def __str__(self):
        return "(P{})".format(self.player)

    def __repr__(self):
        return "({})".format(self.player)
