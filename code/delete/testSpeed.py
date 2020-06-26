import chessModel
from collections import OrderedDict
import numpy as np
import time
import random
import copy
from game import Game
from player import Player
import time
from tensorflow import keras
from pathlib import WindowsPath
from logging import getLogger
import anxilliaryfunctions as af


logger = getLogger(__name__)
inputPath = WindowsPath('C:/Users/Watson/Projects/remus/input/pkl_games/')
model = keras.models.load_model((inputPath / 'watsonBrainNew'))




fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
fenTensor = af.fenToTensor(fen)
inputState = np.expand_dims(fenTensor, axis=0)

t0 = time.time()
for i in range(1500):    
    policy, value = model.predict(inputState)
t1 = time.time()

total_n = t1-t0
print("Cumulative time:", total_n)















def createMoveDict():
    # Create dictionary, label as a key, numpy array as a value
    moveLabels = chessModel.create_uci_labels()
    labelDict = OrderedDict([(i, 0) for i in moveLabels])

    def convertMoves(move, labels):
        labels[move] = 1
        moveTensor = np.array(list(labels.values()))
        labels[move] = 0
        return moveTensor

    return {i: convertMoves(i, labelDict) for i in moveLabels}


b = createMoveDict()

# Test
moveLabels = chessModel.create_uci_labels()
samplelist3 = random.choices(moveLabels, k=10_000_000)


def fenToTensor(input):
    '''
     Define valid fen components. An example:
     fen = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'
     a = fenToTensor(fen)
    '''
    pieces_white = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12}
    pieces_black = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
                    'P': 7, 'N': 8, 'B': 9, 'R': 10, 'Q': 11, 'K': 12}

    # Split fen string into principal components
    splits = input.split()

    # Prepare tensor
    tensor = np.int8(np.zeros((12, 8, 8)))
    rownr = colnr = 0

    # If black has to move, turn the board
    if splits[1] == "b":
        pieces_white = pieces_black

    # Parse the string
    for i, c in enumerate(splits[0]):
        if c in pieces_white:
            tensor[pieces_white[c]-1, rownr, colnr] = 1
            colnr += 1
        elif c == '/':  # new row
            rownr += 1
            colnr = 0
        elif c.isdigit():
            colnr = colnr + int(c)
        else:
            raise ValueError("invalid fen string")

    return tensor


t0 = time.time()

fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
for i in range(1_000_000):
    a = fenToTensor(fen)

t1 = time.time()

total_n = t1-t0
print("Cumulative time:", total_n)


t0 = time.time()
for i in samplelist3:
    # z = convertMoves(samplelist)
    labels = b[i]
t1 = time.time()

total_n = t1-t0
print("Cumulative time:", total_n)


moveLabels2 = chessModel.create_uci_labels()


labelDict = OrderedDict([(i, 0) for i in moveLabels2])

samplelist = random.choices(moveLabels2, k=10000)


# Main test
t0 = time.time()

# values = [convertMoves(x, labelDict) for x in samplelist]
for i in range(1_000_000):
    pass

t1 = time.time()

total_n = t1-t0
print("Cumulative time:", total_n)


# Main test
t0 = time.time()

for i in range(1000000):
    # z = convertMoves(samplelist)
    values = list(labelDict.values())

t1 = time.time()

total_n = t1-t0
print("Cumulative time:", total_n)


t0 = time.time()
for i in range(10000):
    # z = convertMoves(samplelist)
    labels = labelDict.copy()
t1 = time.time()

total_n = t1-t0
print("Cumulative time:", total_n)

t0 = time.time()
for i in range(10000):
    # z = convertMoves(samplelist)
    labels = copy.copy(labelDict)
t1 = time.time()

total_n = t1-t0
print("Cumulative time:", total_n)


samplelist3 = random.choices(moveLabels2, k=2000)

t0 = time.time()
for i in range(10000):
    # z = convertMoves(samplelist)
    labels = list(samplelist3)
t1 = time.time()

total_n = t1-t0
print("Cumulative time:", total_n)
