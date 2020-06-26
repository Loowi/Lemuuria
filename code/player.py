"""
This encapsulates all of the functionality related to actually playing the game itself, not just
making / training predictions.
"""
from logging import getLogger
import chess
import numpy as np
import random
from pathlib import WindowsPath
from tensorflow import keras
import anxilliaryfunctions as af
from enum import Enum
from mcts import MctsTree
import onnxruntime

logger = getLogger(__name__)


class Player:
    """
    Plays the actual game of chess, choosing moves based on policy and value network predictions
    """

    def __init__(self, session=None):
        self.session = session
        self.moves = []


    def createNodes(self, fen):
        root = MctsTree(self.session, fen)
        root.addChildren(root)
        return root
        # root.addChildren(root)

    def mtscMove(self, fen, player, enumDict):
         
        # create root node with children
        # pick the move according to np.random.choice
        # go down, pick move:
            # if no children, create children, update stats and go up
            # if children, pick one and continue until stop criteria is hit
            # if stop criteria is hit, go up and update stats on the way
            # continue as many times as resources permit
        root = self.createNodes(fen)

        for i in range(10):
            print(1, root)
            probs = [child.adjust_value for child in root.children]
            probs = np.array(probs)
            probs /= probs.sum()

            pickChild = np.random.choice(root.children, p=probs)
            print(2, pickChild)
            
        return root.adjust_value








    def moveSimple(self, fen, enumDict):  # Create legitimate moves and pick one
        board = chess.Board(fen)
        if self.session is None:
            moves = list(board.legal_moves)
            move = random.choice(moves)
            self.moves.append(move)
            value = np.nan

            return move, value

        else:        
            fen = af.fenToTensor(fen)
            inputState = np.expand_dims(fen, axis=0)
            inputState = inputState.astype(np.float32)
        
            input_name = self.session.get_inputs()[0].name
            output_name_1 = self.session.get_outputs()[0].name
            output_name_2 = self.session.get_outputs()[1].name
            
            policy, value = self.session.run([output_name_1, output_name_2], {input_name: inputState})

            # policy, value = self.model.predict(inputState)

            # Move with the highest value: find legit moves, max index and lookup in the dict
            moves = [str(x) for x in list(board.legal_moves)]
            kala = {move: policy[0][enumDict[move].value] for move in moves}
            move = max(kala, key=kala.get)
            self.moves.append(move)

            return move, value



# def moveSimple(self, fen, enumDict):  # Create legitimate moves and pick one
#         board = chess.Board(fen)
#         if self.model is None:
#             moves = list(board.legal_moves)
#             move = random.choice(moves)
#             self.moves.append(move)
#             value = np.nan

#             return move, value

#         else:        
#             fen = af.fenToTensor(fen)
#             inputState = np.expand_dims(fen, axis=0)
#             policy, value = self.model.predict(inputState)

#             # Move with the highest value: find legit moves, max index and lookup in the dict
#             moves = [str(x) for x in list(board.legal_moves)]
#             kala = {move: policy[0][enumDict[move].value] for move in moves}
#             move = max(kala, key=kala.get)
#             self.moves.append(move)

#             return move, value

