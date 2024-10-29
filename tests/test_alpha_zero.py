import unittest

from alpha_zero import ResNet 

from tinygrad import Tensor


class TestAlphaZero(unittest.TestCase):
    def test_network_forward_pass_with_batch_size_1(self):
        x = Tensor.rand((1, 17, 19, 19))
        network = ResNet()

        network(x)


if __name__ == "__main__":
    unittest.main()