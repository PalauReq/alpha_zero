from __future__ import annotations

import mcts
import environments.tictactoe as env

from tinygrad import Tensor, nn
from tinygrad.helpers import trange

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def self_learn(num_iterations: int, num_games: int, num_simulations: int):
    f = ResNet((3, 3), 2, 9, 2, env.action_space_size)

    games = []
    for i in range(num_iterations):
        wins, draws, losses = 0, 0, 0
        logger.info(f"Iteration: {i}")
        for j in range(num_games):
            logger.info(f"Game: {j}")
            states, policies, actions, rewards, values = self_play(m, num_simulations)

            match values[0]:
                case 1: wins += 1
                case 0: draws += 1
                case -1: losses += 1

            games.append([states, policies, actions, rewards, values])

        logger.info(f"Player 1: {wins} wins, {draws} draws and {losses} losses")

        # store(games, i)
        x = Tensor([state.board for i, game in enumerate(games) for state in game[0] if i%7 != 0])
        pi = Tensor([policy for i, game in enumerate(games) for policy in game[1] if i%7 != 0])
        v = Tensor([value for i, game in enumerate(games) for value in game[4] if i%7 != 0])
        x_test = Tensor([state.board for i, game in enumerate(games) for state in game[0] if i%7 == 0])
        pi_test = Tensor([policy for i, game in enumerate(games) for policy in game[1] if i%7 == 0])
        v_test = Tensor([value for i, game in enumerate(games) for value in game[4] if i%7 == 0])

        f = optimize(f, (x, pi, v, x_test, pi_test, v_test))
        # nn.state.safe_save(nn.state.get_state_dict(f), f"202502021050_{i:02}.safetensor")

def self_play(f: ResNet, num_simulations: int, v_resign: float = -1):
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
    s = env.State()
    r = None
    node = mcts.MCTNode(None, None, s)
    is_term = False

    states, policies, actions, rewards = [], [], [], []

    while not is_term: # TODO add v > v_resign
        logger.debug(f"Move: {len(states)}")
        pi = mcts.search(node, m, env, num_simulations, temperature=2)
        a, node = mcts.play(node, temperature=2)
        logger.info(f"pi: {pi}, a: {a}")

        states.append(s), policies.append(pi), actions.append(a), rewards.append(r)

        s, r, is_term = env.transition(s, a)
        s.turn()

    # determine the outcome and add z to records
    if r == 0:
        values = [0 for _ in states]
    else:
        values = []
        for i in range(len(states)):
            values.append(r)
            r = -r
        values = values[::-1]

    if actions[0] in [0, 2, 6, 8]:
        logger.info(f"Player 1 did an optimal opening")
        if actions[1] == 4:
            logger.info(f"Player 2 did an optimal move")

    logger.info(f"actions: {actions}")

    return states, policies, actions, rewards, values

    
def optimize(f: ResNet, data, num_steps=1000, batch_size=2048) -> ResNet:
    """
    The batch-size is 32 per worker, for a total mini-batch size of 2,048. Each mini-batch of data
    is sampled uniformly at random from all positions from the most recent 500,000 games of self-play.
    It produces a new checkpoint every 1,000 training steps.
    """
    x, pi, z, x_test, pi_test, z_test  = data
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


class ResNet:
    def __init__(self, board_shape: tuple[int, int] = (19, 19), num_input_planes: int = 17, num_hidden_planes: int = 256, num_residual_blocks: int = 19, action_space_size: int = 362):
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


if __name__ == "__main__":
    self_learn(20, 4, 100)