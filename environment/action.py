class HexAction:

    def __init__(self, player, coord):
        """
        Simple class to represent an action in the game
        :param player: int
        :param cell: Cell
        """
        self.player = player
        self.coord = coord

    def get_player(self):
        return self.player

    def get_coord(self):
        return self.coord

    def __str__(self):
        return "P{}C{}{}".format(self.player, self.coord[0], self.coord[1])

    def __repr__(self):
        return "P{}C{}{}".format(self.player, self.coord[0], self.coord[1])