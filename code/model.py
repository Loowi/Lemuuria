from tensorflow import keras
from pathlib import WindowsPath
from logging import getLogger
import anxilliaryfunctions as af
import numpy as np


logger = getLogger(__name__)


class modelInstance():
    def __init__(self, inputPath, name):
        self.model = keras.models.load_model((inputPath / name))

    def predict(self, state):
        policy, value = self.model.predict(state)
        return policy, value


inputPath = WindowsPath('C:/Users/Watson/Projects/remus/input/pkl_games/')
kala = modelInstance(inputPath, 'watsonBrainNew')


fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
fenTensor = af.fenToTensor(fen)
inputState = np.expand_dims(fenTensor, axis=0)

a, b = kala.predict(inputState)
