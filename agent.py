# Andrew Edwards -- almostimplemented.com
# =======================================
# A checkers agent class.
#
# Last updated: July 21, 2014


class CheckersAgent:
    def __init__(self, move_function, end_function):
        self.move_function = move_function
        self.end_function = end_function

    def make_move(self, board):
        return self.move_function(board)

    def inform_endgame(self, board, lose):
        self.end_function(board, lose)
