class Cell:

    def __init__(self, row, column, is_peg):
        self.row, self.column = row, column  # Location on board
        self.neighbours = {}  # Neighbour cells on board
        self.is_peg = is_peg  # Boolean telling if it is a peg or if it is empty

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

    def __str__(self):
        return "({},{})".format(self.row, self.column)

    def __repr__(self):
        return "({},{},{})".format(self.row, self.column, self.is_peg)
