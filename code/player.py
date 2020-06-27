"""
This encapsulates all of the functionality related to actually playing the game itself, not just
making / training predictions.
"""
from logging import getLogger
import chess
import numpy as np
import random
from pathlib import WindowsPath
import library as lib
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

    def mtscMove(self, fen, enumDict):
        for i in range(200):
            if i == 0:
                root = self.createNodes(fen)
                node = self.createNodes(fen)
                
            # print('Summary:', node.move, node.fen)
            finished, result = self.game_over(node.fen)

            if finished:  # recursion to the top, update values based on result
                # print('End')
                node = self.update_values(node)  # todo: recursion

            elif not node.children:  # Spawn a child and recursion to the top
                # print('No children')
                node.addChildren(node)
                node = self.update_values(node)

            else:  # pick a child and go deeper
                # print('Explore')
                node = self.next_step(node)

        node = self.update_values(node)
        # print('BestMove:', len(node.children), node.move, node.fen)
        move = self.returnBestMove(node.children)
        value = node.mean_value

        return move, value

    def game_over(self, fen):  # place all your conditions here
        board = chess.Board(fen)
        finished = board.is_game_over()
        result = board.result()
        return finished, result

    def next_step(self, node):
        probs = [child.adjust_value for child in node.children]
        probs = np.array(probs)
        probs /= probs.sum()
        node = np.random.choice(node.children, p=probs)
        return node

    def returnBestMove(self, children):
        best_value = -np.inf
        for child in children:
            if child.adjust_value > best_value:  # should be probability based
                best_value = child.adjust_value
                best_move = child.move
        return best_move       

    def update_values(self, node):  # recursion
        if node.parent:
            parent = node.parent
            parent.calcValue(node.total_value)
            return(self.update_values(parent))
        else:
            return node

    def moveSimple(self, fen, enumDict):  # Create legitimate moves and pick one
        board = chess.Board(fen)
        if self.session is None:
            moves = list(board.legal_moves)
            move = random.choice(moves)
            self.moves.append(move)
            value = np.nan

            return move, value

        else:
            fen = lib.fenToTensor(fen)
            inputState = np.expand_dims(fen, axis=0)
            inputState = inputState.astype(np.float32)

            input_name = self.session.get_inputs()[0].name
            output_name_1 = self.session.get_outputs()[0].name
            output_name_2 = self.session.get_outputs()[1].name

            policy, value = self.session.run(
                [output_name_1, output_name_2], {input_name: inputState})

            # Move with the highest value: find legit moves, max index and lookup in the dict
            moves = [str(x) for x in list(board.legal_moves)]
            kala = {move: policy[0][enumDict[move].value] for move in moves}
            move = max(kala, key=kala.get)
            self.moves.append(move)

            return move, value
