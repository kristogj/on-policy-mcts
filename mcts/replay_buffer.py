import random


class ReplayBuffer:

    def __init__(self):
        self.buffer = []

    def get_batch(self, batch_size):
        """
        Return a random mini-batch from the buffer
        :param batch_size: int
        :return: list[tuple(Node, FloatTensor)]
        """
        if batch_size > len(self.buffer):
            return self.buffer
        return random.sample(self.buffer, batch_size)

    def add_case(self, case):
        """
        Append the new case to the buffer
        :param case: tuple(Node, FloatTensor)
        :return: None
        """
        self.buffer.append(case)

    def clear_cache(self):
        """
        Reset the buffer to an empty list
        :return: None
        """
        self.buffer = []
