from collections import defaultdict
import random
import chess
from game import Game
# import anxilliaryfunctions as af
import library as lib
from pathlib import WindowsPath
from enum import Enum
import numpy as np
from logging import getLogger


logger = getLogger(__name__)


class MctsTree:
    """
    The node that stores move and state information required for MCTS
    """

    def __init__(self, session, fen, prior_prob=np.nan, move=None, parent=None):
        self.fen = fen  # chess.Board(fen)
        self.children = []
        self.session = session
        self.parent = parent
        self.move = move

        self.white_to_move = 1

        self.prior_prob = prior_prob
        self.num_visits = 0
        self.total_value = 0
        self.mean_value = prior_prob
        self.adjust_value = self.mean_value + (1.5 * self.prior_prob)

    def __repr__(self):
        s0 = str(self.move)
        s1 = str(self.prior_prob)
        s2 = str(self.num_visits)
        s3 = str(self.adjust_value)
        return 'Move: '+s0+', prior_prob: '+s1+', num_visits: '+s2+', adjust_value: '+s3+' |'

    def calcValue(self, new_value):
        self.num_visits += 1
        self.total_value = np.nansum(new_value + self.total_value)
        self.mean_value = self.total_value / self.num_visits

        add_on = (1.5 * self.prior_prob *
                  np.sqrt(self.num_visits + 1) / (1 + self.num_visits))
        self.adjust_value = self.mean_value + add_on

    def addChildren(self, parent):
        board = chess.Board(self.fen)
        moves = [str(x) for x in list(board.legal_moves)]

        gameState = lib.fenToTensor(self.fen)
        inputState = np.expand_dims(gameState, axis=0)
        inputState = inputState.astype(np.float32)
        
        input_name = self.session.get_inputs()[0].name
        output_name_1 = self.session.get_outputs()[0].name
        output_name_2 = self.session.get_outputs()[1].name
        
        policy, value = self.session.run([output_name_1, output_name_2], {input_name: inputState})

        finalDict, enumDict = lib.createMoveDict()
        values = {move: policy[0][enumDict[move].value] for move in moves}

        self.board_value = value

        for move in values.keys():
            boardTemp = chess.Board(self.fen)
            boardTemp.push(chess.Move.from_uci(move))
            prior_probability = values[move]
            node = MctsTree(self.session, boardTemp.fen(), prior_probability, move)
            self.children.append(node)


 