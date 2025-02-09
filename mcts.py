from __future__ import annotations
from math import sqrt
import logging
import random

from tinygrad import Tensor  # TODO remove MCTS dependency on tinygrad


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCTNode():
    """Represents a state"""
    def __init__(self, parent: MCTNode | None, a: int | None, s, to_play: int = 0, p: float = 0):
        self.n = 0 # visit count
        self.w = 0 # total action-value
        self.q = 0 # mean action-value
        self.p = p # prior probability of selecting the edge

        self.parent = parent
        self.a = a
        self.s = s
        self.to_play = to_play
        self.children = []


    def __repr__(self) -> str:
        return f"MCTNode(n={self.n}, w={self.w:.5f}, q={self.q:.5f}, p={self.p:.5f}, parent={id(self.parent)}, is_leaf={self.is_leaf()}, b={self.s.board.tolist()})"

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_exploring_score(self, sqrt_sum_n: int, c_puct: float = 1) -> float:
        return - self.q + c_puct * self.p * sqrt_sum_n / (1 + self.n)
    
    def get_best_child_to_explore(self, c_puct: float = 1) -> MCTNode:
        sqrt_sum_n = sqrt(sum(child.n for child in self.children))
        return max(self.children, key=lambda x: x.get_exploring_score(sqrt_sum_n, c_puct))
    
    def get_best_child_to_play(self, temperature: float = 1) -> MCTNode:
        sum_n = sum(child.n ** (1 / temperature) for child in self.children)
        scores = [child.n ** (1 / temperature) / sum_n for child in self.children]
        return random.choices(self.children, weights=scores)[0]

    def get_policy(self, temperature: float, action_space: int) -> list[float]:
        sum_n = sum(child.n ** (1 / temperature) for child in self.children)
        sparse_policy = {child.a: child.n ** (1 / temperature) / sum_n for child in self.children}
        return [sparse_policy.get(a, 0) for a in range(action_space)]

    def get_action(self) -> int:
        return self.a


def search(root: MCTNode, f, env, num_simulations: int = 800, temperature: float = 1) -> list[float]:
    for t in range(num_simulations):
        logger.debug(f"simulation: {t}")
        leaf = select(root)
        logger.debug(f"selected leaf: {leaf}")
        expand_and_evaluate(leaf, f, env)
        backup(leaf)

    return root.get_policy(temperature, env.action_space_size)


def select(node: MCTNode, env) -> MCTNode:
    while not node.is_leaf(): # and not env.is_terminal(node.s): # TODO avoid MCTS exploring post terminal states!
        node = node.get_best_child_to_explore()
    return node


def expand_and_evaluate(leaf: MCTNode, f, env):
    # TODO queue nodes for evaluation with batch_size=8
    x = Tensor(leaf.s.board).reshape((1, 2, 3, 3))
    ps, v = f(x) # TODO model should take boards as numpy arrays
    logger.debug(f"s: {leaf.s.board.tolist()}, v: {v.item()}, p: {ps.numpy().tolist()}")
    for a, p in enumerate(ps.numpy().flatten()): # TODO avoid MCTS exploring post terminal states!
        if not env.is_legal(a, leaf.s): continue
        s, _, _ = env.transition(leaf.s, a)
        s.turn()
        child = MCTNode(leaf, a=a, s=s, to_play=(leaf.to_play + 1) % 2, p=p)
        logger.debug(f"adding child: {child}")
        leaf.children.append(child)
    leaf.w = v.item()


def backup(leaf: MCTNode):
    to_play = leaf.to_play
    v = leaf.w
    leaf.n += 1
    leaf.q = leaf.w / leaf.n
    leaf = leaf.parent
    
    while leaf is not None:
        leaf.n += 1
        leaf.w += v if leaf.to_play == to_play else -v
        leaf.q = leaf.w / leaf.n
        leaf = leaf.parent


def play(current_node: MCTNode, temperature: float) -> tuple[int, MCTNode]:
    # TODO use virtual loss to ensure each search thread evaluates different nodes.
    next_node = current_node.get_best_child_to_play(temperature)
    a = next_node.get_action()
    return a, next_node

def print_tree(node: MCTNode, indent: int = 0):
    print(f"{'  ' * indent}{indent} {node}")
    for child in node.children:
        if child.n > 0: print_tree(child, indent + 1)
