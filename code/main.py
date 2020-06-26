from game import Game
from player import Player
import time
from tensorflow import keras
from pathlib import WindowsPath
from logging import getLogger
import onnxruntime


logger = getLogger(__name__)
# inputPath = WindowsPath('C:/Users/Watson/Projects/remus/input/pkl_games/')
# model = keras.models.load_model((inputPath / 'watsonBrainNew'))

session = onnxruntime.InferenceSession('model.onnx', None)

t0 = time.time()
for i in range(1):
    white = Player(session)
    black = Player()
    game = Game(white, black)
    game.play_game(game)
t1 = time.time()

total_n = t1-t0
print("Cumulative time:", total_n)

game.display_pgn()
game.board
