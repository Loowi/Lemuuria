from game import Game
from player import Player
import time
from pathlib import Path
from logging import getLogger
import onnxruntime
import chess

logger = getLogger(__name__)

inputPath = Path('C:/Users/Watson/Projects/lemuuria/input/')
session = onnxruntime.InferenceSession(str(inputPath / 'model.onnx'), None)

startBoard = chess.Board('8/p4pk1/4n1p1/1p2P2p/q4P1P/P4QP1/5BK1/8 w - - 0 1')

t0 = time.time()
for i in range(1):
    white = Player(session)
    black = Player()
    game = Game(white, black, startBoard)
    game.play_game()
t1 = time.time()

total_n = t1-t0
print("Cumulative time:", total_n)


# # python -m cProfile -s tottime main.py
# import pickle
# with open('node_999.pkl', 'rb') as f:
#     x = pickle.load(f)


# def update_values(node):  # recursion
#     if node.parent:
#         parent = node.parent
#         parent.calcValue(node.board_value)
#         return(update_values(parent))
#     else:
#         return node

# a2 = update_values(x)