import unittest

from mcts import *


class TestMCTS(unittest.TestCase):
    def test_search(self):
        class ConstantModel():
            def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
                return Tensor([0.25]*4), Tensor([0.1])

        model = ConstantModel()
        import environments.tictactoe as env
        root = MCTNode(parent=None, a=None, s=env.State())
        
        policy = search(root, model, env)
        print(policy)


if __name__ == "__main__":
    unittest.main()