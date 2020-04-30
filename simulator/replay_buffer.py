import random
from collections import deque
import numpy as np


class ReplayBuffer:

    def __init__(self):
        self.buffer = deque()
        self.max_len = 800

    def get_batch(self, batch_size):
        """
        Return a random mini-batch from the buffer
        :param batch_size: int
        :return: list[tuple(Node, FloatTensor)]
        """
        if batch_size > len(self.buffer):
            return self.buffer

        # Do a weighted random choice, where newer cases are prioritized over older cases
        weights = np.linspace(0, 1, len(self.buffer))
        return random.choices(self.buffer, weights=weights, k=batch_size)

    def add_case(self, case):
        """
        Append the new case to the buffer
        :param case: tuple(Node, FloatTensor)
        :return: None
        """
        if len(self.buffer) > self.max_len:
            self.buffer.popleft()
        self.buffer.append(case)

    def clear_cache(self):
        """
        Reset the buffer to an empty list
        :return: None
        """
        self.buffer = []
