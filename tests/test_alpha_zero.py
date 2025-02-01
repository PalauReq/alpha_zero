import unittest

from alpha_zero import *

from tinygrad import Tensor
import numpy as np


class TestAlphaZero(unittest.TestCase):
    def test_network_forward_pass_with_batch_size_1(self):
        x = Tensor.randint((1, 17, 19, 19), low=0, high=1)
        network = ResNet()

        ps, v = network(x)

        self.assertEqual(ps.shape, (1, 362))
        self.assertEqual(v.shape, (1, 1))


    def test_network_forward_pass_with_batch_size_8(self):
        x = Tensor.randint((8, 17, 19, 19), low=0, high=1)
        network = ResNet()

        ps, v = network(x)

        self.assertEqual(ps.shape, (8, 362))
        self.assertEqual(v.shape, (8, 1))


    def test_optimization_loss_function(self):
        x = Tensor.randint((8, 17, 19, 19), low=0, high=1)
        pi = Tensor.rand((8, 362))
        z = Tensor.randint((8, ), low=-1, high=1)
        network = ResNet()

        ps, v = network(x)
        loss_policy = ps.cross_entropy(pi, reduction="none")
        loss_value = (v.squeeze() - z) ** 2
        loss = (ps.cross_entropy(pi, reduction="none") + (v.squeeze() - z) ** 2)

        self.assertEqual(ps.shape, (8, 362))
        self.assertEqual(v.shape, (8, 1))
        self.assertEqual(loss_policy.shape, (8, ))
        self.assertEqual(loss_value.shape, (8, ))
        self.assertEqual(loss.shape, (8, ))
        self.assertGreater(loss.mean().item(), 0)


    def test_optimize(self):
        import environments.tictactoe as env
        import random

        def simulate_games(num_games: int) -> tuple[list[np.ndarray], list[int], list[float]]:
            x, pi, z = [], [], []
            for i in range(num_games):
                states, actions, values = simulate_game()
                x.extend(states), pi.extend(actions), z.extend(values)
            return x, pi, z


        def simulate_game() -> tuple[list[np.ndarray], list[int], list[float]]:
            is_term = False
            s = env.State()
            states = []
            actions = []
            while not is_term:
                states.append(s.board)

                legal_actions = env.get_legal_actions(s)
                a = random.choice(legal_actions)
                actions.append(a)

                s, r, is_term = env.transition(s, a)
                s.turn()

            if r == 0:
                values = [0 for _ in states]
            else:
                values = []
                for i in range(len(states)):
                    values.append(r)
                    r = -r
                values = values[::-1]

            return states, actions, values


        x, pi, z = (Tensor(a) for a in simulate_games(4))
        x_test, pi_test, z_test = (Tensor(a) for a in simulate_games(1))
        f = ResNet((3, 3), 2, 4, 2)
        _ = optimize(f, (x, pi, z, x_test, pi_test, z_test), num_steps=2, batch_size=8)

    def test_self_play(self):
        f = ResNet((3, 3), 2, 4, 2)
        states, policies, actions, rewards, values = self_play(f, num_simulations=10)
        for state in states:
            print(f"state: {state}")
        # print(f"policies: {policies}")
        print(f"actions: {actions}")
        print(f"rewards: {rewards}")
        print(f"values: {values}")

    def test_self_learn(self):
       self_learn(num_iterations=10, num_games=5, num_simulations=20)

if __name__ == "__main__":
    unittest.main()
