from tinygrad import Tensor, nn


class Network:
    def __init__(self, board_shape: tuple[int, int] = (19, 19), num_input_planes: int = 17, num_hidden_planes: int = 256, num_residual_blocks: int = 19):
        board_size = board_shape[0] * board_shape[1]

        self.blocks = [
            ConvolutionalBlock(num_input_planes, num_hidden_planes),
            *[ResidualBlock(num_hidden_planes) for _ in range(num_residual_blocks)],
            ]
        self.policy_head = [
            nn.Conv2d(num_hidden_planes, 2, (1, 1)), nn.BatchNorm(2), Tensor.relu,
            lambda x: x.view(-1, board_size * 2), nn.Linear(board_size * 2, board_size + 1),
            ]
        self.value_head = [
            nn.Conv2d(num_hidden_planes, 1, (1, 1)), nn.BatchNorm(1), Tensor.relu,
            lambda x: x.view(-1, board_size), nn.Linear(board_size, num_hidden_planes), Tensor.relu,
            nn.Linear(num_hidden_planes, 1), Tensor.tanh,
            ]

    def __call__(self, x: Tensor) -> Tensor:
        y = x.sequential(self.blocks)
        return y.sequential(self.policy_head), y.sequential(self.value_head)


class ConvolutionalBlock:
    def __init__(self, num_input_planes: int, num_output_planes: int): 
        self.layers = [nn.Conv2d(num_input_planes, num_output_planes, (3, 3), padding=1), nn.BatchNorm(num_output_planes), Tensor.relu]

    def __call__(self, x: Tensor) -> Tensor:
        return x.sequential(self.layers)


class ResidualBlock:
    def __init__(self, num_planes: int):
        self.layers = [
            nn.Conv2d(num_planes, num_planes, (3, 3), padding=1), nn.BatchNorm(num_planes), Tensor.relu,
            nn.Conv2d(num_planes, num_planes, (3, 3), padding=1), nn.BatchNorm(num_planes),
        ]
    
    def __call__(self, x: Tensor) -> Tensor:
        return (x.sequential(self.layers) + x).relu()