from __future__ import annotations

import mcts

import numpy as np
from tinygrad import Tensor, nn


def self_learn(num_iterations: int, num_games: int, num_simulations: int):
    f = ResNet()
    
    for i in range(num_iterations):
        games = (self_play(f, num_simulations) for j in range(num_games))
        store(games, i)
        data = np.ndarray(move for game in games for move in game)

        f = optimize(f, data)
        checkpoint(f, i)


def self_play(f: Model, num_simulations: int, v_resign: float = -1) -> np.ndarray:
    """
    For the first 30 moves of each game, the temperature is set to τ = 1; this selects moves proportionally
    to their visit count in MCTS, and ensures a diverse set of positions are encountered. For the remainder
    of the game, an infinitesimal temperature is used, τ → 0. Additional exploration is achieved by adding 
    Dirichlet noise to the prior probabilities in the root node s0, this noise ensures that all moves may be
    tried, but the search may still overrule bad moves. 

    In order to save computation, clearly lost games are resigned. The resignation threshold vresign is selected
    automatically to keep the fraction of false positives (games that could have been won if AlphaGo had not
    resigned) below 5%. To measure false positives, we disable resignation in 10% of self-play games and play
    until termination.
    """
    env = Environment()
    s = State()

    while not env.is_terminal(s) and v > v_resign:
        v, ps = mcts.search(node, f, env, num_simulations)
        record(s, ps)  # search should return these
        a, node = mcts.play(node)

    # determine the outcome and add z to records

    
def optimize(f: Model, data: np.ndarray) -> Model:
    """
    The batch-size is 32 per worker, for a total mini-batch size of 2,048. Each mini-batch of data
    is sampled uniformly at random from all positions from the most recent 500,000 games of self-play.
    It produces a new checkpoint every 1,000 training steps.
    """    
    X_train, Y_train, X_test, Y_test = mnist()
    print("*** got data")

    model = Model()
    print("*** got model")

    opt = nn.optim.Adam(nn.state.get_parameters(model))
    print("*** got optimizer")

    samples = Tensor.randint(getenv("STEPS", 10), getenv("BS", 512), high=X_train.shape[0])
    X_samp, Y_samp = X_train[samples], Y_train[samples]
    print("*** got samples")

    with Tensor.train():
        # TODO: this shouldn't be a for loop. something like: (contract is still up in the air)
        """
        i = UOp.range(samples.shape[0])  # TODO: fix range function on UOp
        losses = model(X_samp[i]).sparse_categorical_crossentropy(Y_samp[i]).backward().contract(i)
        opt.schedule_steps(i)
        """
        losses = []
        for i in range(samples.shape[0]):
        opt.zero_grad()
        losses.append(model(X_samp[i]).sparse_categorical_crossentropy(Y_samp[i]).backward())
        opt.schedule_step()
        # TODO: this stack currently breaks the "generator" aspect of losses. it probably shouldn't
        #losses = Tensor.stack(*losses)
    print("*** scheduled training")

    # evaluate the model
    with Tensor.test():
        test_acc = ((model(X_test).argmax(axis=1) == Y_test).mean()*100)
    print("*** scheduled eval")

    # NOTE: there's no kernels run in the scheduling phase
    assert GlobalCounters.kernel_count == 0, "kernels were run during scheduling!"

    # only actually do anything at the end
    if getenv("LOSS", 1):
        for i in (t:=trange(len(losses))): t.set_description(f"loss: {losses[i].item():6.2f}")
    print(f"test_accuracy: {test_acc.item():5.2f}%")



class ResNet:
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