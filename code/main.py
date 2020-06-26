from game import Game
from player import Player
import time
from pathlib import Path
from logging import getLogger
import onnxruntime


logger = getLogger(__name__)
inputPath = Path('C:/Users/Watson/Projects/lemuuria/input/')
session = onnxruntime.InferenceSession(str(inputPath / 'model.onnx'), None)

t0 = time.time()
for i in range(1):
    white = Player(session)
    black = Player()
    game = Game(white, black)
    game.play_game(game)
t1 = time.time()

total_n = t1-t0
print("Cumulative time:", total_n)
