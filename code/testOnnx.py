from game import Game
from player import Player
import time
from pathlib import Path
from logging import getLogger
import onnxruntime
import chess
import library as lib
import numpy as np

logger = getLogger(__name__)

inputPath = Path('C:/Users/Watson/Projects/lemuuria/input/')
session = onnxruntime.InferenceSession(str(inputPath / 'model.onnx'), None)

fen = '8/p4pk1/4n1p1/1p2P2p/5P1P/q4QPK/5B2/8 w - - 0 2'

gameState = lib.fenToTensor(fen)
inputState = np.expand_dims(gameState, axis=0)
inputState = inputState.astype(np.float32)

input_name = session.get_inputs()[0].name
output_1 = session.get_outputs()[0].name
output_2 = session.get_outputs()[1].name

policy, value = session.run([output_1, output_2], {input_name: inputState})

print(value)

board = chess.Board(fen)
board