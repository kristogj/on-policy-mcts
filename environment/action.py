class HexAction:

    def __init__(self, player, cell):
        """
        Simple class to represent an action in the game
        :param player: int
        :param cell: Cell
        """
        self.player = player
        self.cell = cell

    def get_player(self):
        return self.player

    def get_cell(self):
        return self.cell

    def __str__(self):
        return "P{}C{}{}".format(self.player, self.cell.row, self.cell.column)

    def __repr__(self):
        return "P{}C{}{}".format(self.player, self.cell.row, self.cell.column)