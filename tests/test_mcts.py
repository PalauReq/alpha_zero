import unittest

from mcts import *
import environments.tictactoe as env
from alpha_zero import Model, ResNet

import numpy as np


class TestMCTS(unittest.TestCase):
    class MockModel(Model):
        def __call__(self, x: np.ndarray) -> tuple[np.ndarray, float]:
            return np.array([1/9] * 9), 0

    model = MockModel(ResNet((3, 3), 2, 1, 1, 9))

    def test_search_winner(self):
        board = np.array([[[0, 0, 0], [1, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [1, 1, 0]]])
        node = Node(parent=None, a=None, s=env.State(board))
        policy = search(node, self.model, env, 10)
        self.assertEqual(policy.index(max(policy)), 5)

    def test_search_blocker(self):
        board = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [1, 1, 0]]])
        node = Node(parent=None, a=None, s=env.State(board))
        policy = search(node, self.model, env, 40)
        self.assertEqual(policy.index(max(policy)), 8)

    def test_search_winner_in_three(self):
        board = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 1, 1], [0, 0, 0], [0, 0, 0]]])
        node = Node(parent=None, a=None, s=env.State(board))
        policy = search(node, self.model, env, 100)
        self.assertEqual(policy.index(max(policy)), 6)


if __name__ == "__main__":
    unittest.main()