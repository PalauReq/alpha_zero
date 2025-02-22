import unittest

from alpha_zero import *
from mcts import print_tree

import numpy as np


class TestAlphaZero(unittest.TestCase):
    class EnvironmentModel():
        def __call__(self, x: np.ndarray) -> tuple[np.ndarray, float]:
            r = env.compute_reward(env.State(x))
            return np.array([1/9] * 9), r

    class MockModel():
        def __call__(self, x: np.ndarray) -> tuple[np.ndarray, float]:
            r = env.compute_reward(env.State(x))
            return np.array([1/9] * 9), r


    def test_win_in_one(self):
        f = self.MockModel()
        board = np.array([[[0, 0, 0], [0, 0, 0], [0, 1, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 0]]])
        node = mcts.Node(parent=None, a=None, s=env.State(board))
        pi = mcts.search(node, f, env, num_simulations=1000)
        print_tree(node)
        print(f"pi: {pi}")
        a, node = mcts.play(node, temperature=2)
        print(f"a: {a}, node: {node}")


    def test_self_play(self):
        f = self.EnvironmentModel()
        states, policies, actions, rewards, values = self_play(f, num_simulations=10000)
        for state in states:
            print(f"state: {state}")
        for policy in policies:
            print(f"policies: {policy}")
        print(f"actions: {actions}")
        print(f"rewards: {rewards}")
        print(f"values: {values}")

    @unittest.skip("Too many num_steps and batch_size too large")
    def test_self_learn(self):
       self_learn(num_iterations=100, num_games=8, num_simulations=100)
if __name__ == "__main__":
    unittest.main()
