import unittest

from environments.tictactoe import *


class TestTicTacToe(unittest.TestCase):
    def test_make_move(self):
        s = State().make_move(0, 0, 0)
        self.assertEqual(s.board[0, 0, 0], 1)

    def test_is_cell_available(self):
        s = State()
        self.assertTrue(s.is_cell_available(0, 0))
        
        new_s = s.make_move(0, 0, 0)
        self.assertFalse(new_s.is_cell_available(0, 0))

    def test_is_winner(self):
        # Horizontal win for player 0
        s = State().make_move(0, 0, 0).make_move(0, 1, 0).make_move(0, 2, 0)
        self.assertTrue(s.is_winner(0))

        # Vertical win for player 1
        s = State().make_move(0, 0, 1).make_move(1, 0, 1).make_move(2, 0, 1)
        self.assertTrue(s.is_winner(1))

        # Diagonal win for player 0
        s = State().make_move(0, 0, 0).make_move(1, 1, 0).make_move(2, 2, 0)
        self.assertTrue(s.is_winner(0))

        # Anti-diagonal win for player 1
        s = State().make_move(0, 2, 1).make_move(1, 1, 1).make_move(2, 0, 1)
        self.assertTrue(s.is_winner(1))

    def test_is_full(self):
        s = State()
        for i in range(num_rows):
            for j in range(num_cols):
                s = s.make_move(i, j, 0)
        self.assertTrue(s.is_full())

    def test_transition(self):
        new_s, reward, done = transition(State(), 0)
        
        self.assertEqual(new_s.board[0, 0, 0], 1)
        self.assertEqual(reward, 0)
        self.assertFalse(done)

    def test_is_legal(self):
        s = State()
        self.assertTrue(is_legal(0, s))
        self.assertFalse(is_legal(9, s))  # Out of bounds

        s = s.make_move(0, 0, 0)
        self.assertFalse(is_legal(0, s))  # Cell already occupied

    def test_compute_reward(self):
        # Player 0 wins
        s = State().make_move(0, 0, 0).make_move(0, 1, 0).make_move(0, 2, 0)
        self.assertEqual(compute_reward(s), 1)

        # Player 1 wins
        s = State().make_move(0, 0, 1).make_move(1, 0, 1).make_move(2, 0, 1)
        self.assertEqual(compute_reward(s), -1)

        # Draw
        s = State()
        self.assertEqual(compute_reward(s), 0)

        # Fill the board without a winner
        s = (State()
             .make_move(1, 1, 0).make_move(0, 0, 1)
             .make_move(2, 0, 0).make_move(0, 2, 1)
             .make_move(0, 1, 0).make_move(2, 1, 1)
             .make_move(1, 2, 0).make_move(1, 0, 1)
             .make_move(2, 2, 0))
        self.assertEqual(compute_reward(s), 0)

    def test_is_terminal(self):
        # Player 0 wins
        s = State().make_move(0, 0, 0).make_move(0, 1, 0).make_move(0, 2, 0)
        self.assertTrue(is_terminal(s))

        # Player 1 wins
        s = State().make_move(0, 0, 1).make_move(1, 0, 1).make_move(2, 0, 1)
        self.assertTrue(is_terminal(s))

        # Draw
        # Fill the board without a winner
        s = (State()
             .make_move(1, 1, 0).make_move(0, 0, 1)
             .make_move(2, 0, 0).make_move(0, 2, 1)
             .make_move(0, 1, 0).make_move(2, 1, 1)
             .make_move(1, 2, 0).make_move(1, 0, 1)
             .make_move(2, 2, 0))
        self.assertTrue(is_terminal(s))

if __name__ == "__main__":
    unittest.main()