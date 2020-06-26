
import onnxruntime
import anxilliaryfunctions as af
from onnxruntime.datasets import get_example
import numpy as np
import onnxruntime as rt
import onnxmltools
from game import Game
from player import Player
import time
from tensorflow import keras
from pathlib import WindowsPath
from logging import getLogger


# logger = getLogger(__name__)
# inputPath = WindowsPath('C:/Users/Watson/Projects/remus/input/pkl_games/')
# keras_model = keras.models.load_model((inputPath / 'watsonBrainNew'))

# # Change this path to the output name and path for the ONNX model
# output_onnx_model = 'model.onnx'

# # Convert the Keras model into ONNX
# onnx_model = onnxmltools.convert_keras(keras_model)

# # Save as protobuf
# onnxmltools.utils.save_model(onnx_model, output_onnx_model)


# # example1 = get_example('model.onnx')
# # sess = rt.InferenceSession(example1)

# Run the model on the backend
session = rt.InferenceSession('model.onnx', None)


# get the name of the first input of the model
fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
fenTensor = af.fenToTensor(fen)
inputState = np.expand_dims(fenTensor, axis=0)
inputState = inputState.astype(np.float32)

session = onnxruntime.InferenceSession('model.onnx', None)
input_name = session.get_inputs()[0].name
output_name_1 = session.get_outputs()[0].name
output_name_2 = session.get_outputs()[1].name

t0 = time.time()
for i in range(150):

    policy, value = session.run([output_name_1, output_name_2], {input_name: inputState})

t1 = time.time()

total_n = t1-t0
print("Cumulative time:", total_n)


# t = session.get_inputs()[0]
