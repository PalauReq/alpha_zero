from tinygrad import Tensor, nn


class Network:
    def __init__(self, board_shape: tuple[int, int] = (19, 19), num_planes: int = 17):
        board_size = board_shape[0] * board_shape[1]

        self.blocks = [ConvolutionalBlock(num_planes)] + [ResidualBlock() for _ in range(19)]
        self.policy_head = [
            nn.Conv2d(256, 2, (1, 1)), nn.BatchNorm(2), Tensor.relu, 
            lambda x: x.view(-1, board_size * 2), nn.Linear(board_size * 2, board_size + 1),
            ]
        self.value_head = [
            nn.Conv2d(256, 1, (1, 1)), nn.BatchNorm(1), Tensor.relu, 
            lambda x: x.view(-1, board_size), nn.Linear(board_size, 256), Tensor.relu, 
            nn.Linear(256, 1), Tensor.tanh,
            ]

    def __call__(self, x: Tensor) -> Tensor:
        y = x.sequential(self.blocks)
        return y.sequential(self.policy_head), y.sequential(self.value_head)


class ConvolutionalBlock:
    def __init__(self, num_planes: int = 17): 
        self.layers = [nn.Conv2d(num_planes, 256, (3, 3), padding=1), nn.BatchNorm(256), Tensor.relu]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


class ResidualBlock:
    def __init__(self):
        self.layers = [
            nn.Conv2d(256, 256, (3, 3), padding=1), nn.BatchNorm(256), Tensor.relu,
            nn.Conv2d(256, 256, (3, 3), padding=1), nn.BatchNorm(256),
        ]
    
    def __call__(self, x: Tensor) -> Tensor:
        return (x.sequential(self.layers) + x).relu()