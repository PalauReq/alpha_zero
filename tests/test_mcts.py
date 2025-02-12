import unittest

from mcts import *
import environments.tictactoe as env

import numpy as np
from tinygrad import Tensor


class TestMCTS(unittest.TestCase):
    class MockModel():
        def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
            return Tensor([1/9] * 9), Tensor(0)

    model = MockModel()

    def test_search_winner(self):
        board = np.array([[[0, 0, 0], [1, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [1, 1, 0]]])
        node = MCTNode(parent=None, a=None, s=env.State(board))
        policy = search(node, self.model, env, 10)
        self.assertEqual(policy.index(max(policy)), 5)

    def test_search_blocker(self):
        board = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [1, 1, 0]]])
        node = MCTNode(parent=None, a=None, s=env.State(board))
        policy = search(node, self.model, env, 40)
        self.assertEqual(policy.index(max(policy)), 8)

    def test_search_winner_in_three(self):
        board = np.array([[[1, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 1, 1], [0, 0, 0], [0, 0, 0]]])
        node = MCTNode(parent=None, a=None, s=env.State(board))
        policy = search(node, self.model, env, 100)
        self.assertEqual(policy.index(max(policy)), 6)


if __name__ == "__main__":
    unittest.main()