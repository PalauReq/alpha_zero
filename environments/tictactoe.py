from __future__ import annotations
from dataclasses import dataclass 

import numpy as np


num_players = 2
num_rows, num_cols = 3, 3


@dataclass()
class State():
    def __init__(self, board: np.ndarray = np.zeros((num_players, num_rows, num_cols), dtype=int)):
        self.board = board

    def make_move(self, row: int, col: int, player: int) -> State:
        assert self.is_cell_available(row, col), "Cell is not available!"

        new_board = self.board.copy()
        new_board[player, row, col] = 1
        return State(new_board)
    
    def is_cell_available(self, i: int, j: int) -> bool:
        return (self.board[:, i, j] == 0).all()

    def is_winner(self, player: int) -> bool:
        if any((self.board[player, i, :] == 1).all() for i in range(num_rows)): return True
        if any((self.board[player, :, j] == 1).all() for j in range(num_cols)): return True
        if (self.board[player].diagonal() == 1).all(): return True
        if (np.fliplr(self.board[player]).diagonal() == 1).all(): return True
        return False
    
    def is_full(self) -> bool:
        return ((self.board == 1).any(axis=0) == 1).all()
    
    def turn(self):
        self.board = np.flip(self.board, axis=0)
    
    
def transition(s: State, a: int) -> tuple[State, float, bool]:
    i, j = a // num_rows, a % num_rows
    assert _is_legal(i, j, s), "Ilegal action"
    new_s = s.make_move(i, j, player=0)

    return (new_s, compute_reward(new_s), is_terminal(new_s)) # Maybe a similar function just returning new_s is handy


def is_legal(a: int, s: State) -> bool:
    return _is_legal(a // num_rows, a % num_rows , s)


def _is_legal(i: int, j: int, s: State) -> bool:
    if i >= num_rows or j >= num_cols: return False
    if not s.is_cell_available(i, j): return False
    return True


def compute_reward(s: State) -> float:
    if s.is_winner(player=0): return 1
    if s.is_winner(player=1): return -1
    return 0


def is_terminal(s: State) -> bool:
    return s.is_full() or s.is_winner(player=0) or s.is_winner(player=1)
