class Node:

    def __init__(self, state, action, player=None):
        self.state = state
        self.player = player
        self.action = action
        self.parent = None
        self.children = []

        # Values that get updated through backward propagation of the MCTS
        self.value = 0
        self.total = 0

    def increase_total(self):
        """
        Node was on the path taken in the search - increase total
        :return:
        """
        self.total += 1

    def set_parent(self, node):
        """
        Set the parent for this node equal to node
        :param node: Node
        :return:
        """
        self.parent = node

    def set_action(self, action):
        """
        Set the action taken to get to this node
        :param action: Type depends on the game
        :return: None
        """
        self.action = action

    def get_children(self):
        """
        Return the children for this node
        :return: list[Node]
        """
        return self.children

    def __str__(self):
        return "State: {} Action: {} Win: {} Total: {}".format(self.state, self.action, self.win, self.total)
