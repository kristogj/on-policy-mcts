class HexAction:

    def __init__(self, player: int, coord: tuple):
        """
        Simple class to represent an action in the game
        :param player: int
        :param coord: tuple
        """
        self.player = player
        self.coord = coord

    def get_player(self) -> int:
        """
        Return the player making the action
        :return: the player id
        """
        return self.player

    def get_coord(self) -> tuple:
        """
        Return the coordinate for the action
        :return: coordinate for the action
        """
        return self.coord

    def __str__(self):
        return "P{}C{}{}".format(self.player, self.coord[0], self.coord[1])

    def __repr__(self):
        return "P{}C{}{}".format(self.player, self.coord[0], self.coord[1])