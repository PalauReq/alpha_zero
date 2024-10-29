from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from math import sqrt


@dataclass
class State():
    data: list


class Environment():
    def transition(self, s: State, a: int) -> State:
        return State    


class Model():
    @abstractmethod
    def __call__(self, s: State) -> tuple[list[float], float]:
        pass


class MCTNode():
    """Represents a state"""
    def __init__(self, parent: MCTNode, s: State, p: float = 0): 
        self.n = 0 # visit count
        self.w = 0 # total action-value
        self.q = 0 # mean action-value
        self.p = p # prior probability of selecting the edge

        self.parent = parent
        self.children = [] # states if tranistioning from self with action=index

        self.s = s

    def __repr__(self) -> str:
        return f"MCTNode(n={self.n}, w={self.w}, q={self.q}, p={self.p}, parent={id(self.parent)}, is_leaf={self.is_leaf()})"
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def get_best_child_to_explore(self, c_puct: float = 1) -> MCTNode:
        sqrt_sum_n = sqrt(sum(child.n for child in self.children))
        return max(self.children, key=lambda x: x.q + x.upper_confidence_bound(sqrt_sum_n, c_puct))
    
    def upper_confidence_bound(self, sqrt_sum_n: float, c_puct: float = 1) -> float:
        return c_puct * self.p * sqrt_sum_n / (1 + self.n)
    
    def get_best_child_to_play(self, temperature: float = 1) -> MCTNode:
        sum_n = sum(child.n for child in self.children) ** (1 / temperature)
        return max(self.children, key=lambda x: x.n ** (1 / temperature) / sum_n)

    def get_action(self, child: MCTNode) -> int:
        return self.children.index(child)


def search(root: MCTNode, f: Model, env: Environment, num_simulations: int = 800) -> tuple[MCTNode, int]:
    for t in range(num_simulations):
        leaf = select(root)
        expand_and_evaluate(leaf, f, env)
        backup(leaf)
    
    return play(root)


def select(node: MCTNode) -> MCTNode:
    while not node.is_leaf():
        node = node.get_best_child_to_explore()
    return node


def expand_and_evaluate(leaf: MCTNode, f: Model, env: Environment):
    # TODO queue nodes for evaluation with batch_size=8
    ps, v = f(leaf.s) # TODO implement state representation as Tensor
    leaf.children = [MCTNode(leaf, env.transition(leaf.s, a), p) for a, p in enumerate(ps)]
    leaf.w = v


def backup(leaf: MCTNode):
    v = leaf. w
    leaf.n += 1
    leaf.q = leaf.w / leaf.n
    leaf = leaf.parent
    
    while leaf is not None:
        leaf.n += 1
        leaf.w += v
        leaf.q = leaf.w / leaf.n
        leaf = leaf.parent


def play(current_node: MCTNode, temperature: float = 1) -> tuple[MCTNode, int]:
    # TODO use virtual loss to ensure each seach thread evaluates different nodes (69).
    next_node = current_node.get_best_child_to_play(temperature)
    a = current_node.get_action(next_node)
    return a, next_node