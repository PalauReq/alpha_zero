from __future__ import annotations

import numpy as np
from tinygrad import Tensor, nn
from tinygrad.helpers import trange


class Model:
    def __init__(self, f: ResNet):
        self.f = f

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        # TODO queue nodes for evaluation with batch_size=8
        ps, v = self.f(Tensor(x).unsqueeze(0))  # x has not a batch dimension
        return ps.numpy().flatten(), v.item()


class ResNet:
    def __init__(self, board_shape: tuple[int, int] = (19, 19), num_input_planes: int = 17,
                 num_hidden_planes: int = 256, num_residual_blocks: int = 19, action_space_size: int = 362):
        board_size = board_shape[0] * board_shape[1]

        self.blocks = [
            ConvolutionalBlock(num_input_planes, num_hidden_planes),
            *[ResidualBlock(num_hidden_planes) for _ in range(num_residual_blocks)],
        ]
        self.policy_head = [
            nn.Conv2d(num_hidden_planes, 2, (1, 1)), nn.BatchNorm(2), Tensor.relu,
            lambda x: x.view(-1, board_size * 2), nn.Linear(board_size * 2, action_space_size),
        ]
        self.value_head = [
            nn.Conv2d(num_hidden_planes, 1, (1, 1)), nn.BatchNorm(1), Tensor.relu,
            lambda x: x.view(-1, board_size), nn.Linear(board_size, num_hidden_planes), Tensor.relu,
            nn.Linear(num_hidden_planes, 1), Tensor.tanh,
        ]

    def __call__(self, x: Tensor) -> tuple[Tensor, Tensor]:
        y = x.sequential(self.blocks)
        return y.sequential(self.policy_head), y.sequential(self.value_head)


class ConvolutionalBlock:
    def __init__(self, num_input_planes: int, num_output_planes: int):
        self.layers = [nn.Conv2d(num_input_planes, num_output_planes, (3, 3), padding=1),
                       nn.BatchNorm(num_output_planes), Tensor.relu]

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


def optimize(f: ResNet, data: tuple, num_steps=1000, batch_size=2048) -> ResNet:
    """
    The batch-size is 32 per worker, for a total mini-batch size of 2,048. Each mini-batch of data
    is sampled uniformly at random from all positions from the most recent 500,000 games of self-play.
    It produces a new checkpoint every 1,000 training steps.
    """
    x, pi, z, x_test, pi_test, z_test  = (Tensor(d) for d in data)
    samples = Tensor.randint((num_steps, batch_size), high=x.shape[0])
    x, pi, z = x[samples], pi[samples], z[samples]
    opt = nn.optim.Adam(nn.state.get_parameters(f))

    with Tensor.train():
        losses = []
        for i in range(samples.shape[0]):
            opt.zero_grad()
            ps, v = f(x[i])
            # cross_entropy expects target of shape (batch_size,) and returns (batch_size,)
            loss = (ps.cross_entropy(pi[i], reduction="none") + (v.squeeze() - z[i]) ** 2).mean(axis=0)
            losses.append(loss.backward())
            opt.schedule_step()

    with Tensor.test():
        ps, v = f(x_test)
        test_loss = (ps.cross_entropy(pi_test, reduction="none") + (v.squeeze() - z_test) ** 2).mean(axis=0)

    for i in (t:=trange(len(losses))): t.set_description(f"loss: {losses[i].item():6.2f}")
    print(f"test_loss: {test_loss.item():5.2f}")
    return f
