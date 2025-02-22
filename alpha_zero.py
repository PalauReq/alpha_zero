from __future__ import annotations

import mcts
from model import Model, ResNet, optimize
import environments.tictactoe as env

import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def self_learn(num_iterations: int, num_games: int, num_simulations: int):
    f = ResNet((3, 3), 2, 9, 2, env.action_space_size)

    games = []
    for i in range(num_iterations):
        wins, draws, losses = 0, 0, 0
        m = Model(f)
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
        x = [state.board for i, game in enumerate(games) for state in game[0] if i%7 != 0]
        pi = [policy for i, game in enumerate(games) for policy in game[1] if i%7 != 0]
        v = [value for i, game in enumerate(games) for value in game[4] if i%7 != 0]
        x_test = [state.board for i, game in enumerate(games) for state in game[0] if i%7 == 0]
        pi_test = [policy for i, game in enumerate(games) for policy in game[1] if i%7 == 0]
        v_test = [value for i, game in enumerate(games) for value in game[4] if i%7 == 0]

        f = optimize(f, (x, pi, v, x_test, pi_test, v_test))
        # nn.state.safe_save(nn.state.get_state_dict(f), f"202502021050_{i:02}.safetensor")

def self_play(m: Model, num_simulations: int, v_resign: float = -1):
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
    node = mcts.Node(None, None, s)
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

    
if __name__ == "__main__":
    self_learn(20, 4, 100)