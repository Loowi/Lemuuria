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
import pickle


logger = getLogger(__name__)


class Player:
    """
    Plays the actual game of chess, choosing moves based on policy and value network predictions
    """

    def __init__(self, session=None):
        self.session = session
        self.moves = []

    def createNodes(self, fen, enumDict):
        root = MctsTree(12, fen) # todo
        root.addChildren(root, enumDict)
        return root

    def mtscMove(self, fen, enumDict):
        for i in range(300000):
            if i == 0:
                root = self.createNodes(fen, enumDict)
                node = self.createNodes(fen, enumDict)
                # kala1 = [child.adjust_value for child in node.children]
                # print(kala1)
                
            # print('Summary:', node.move, node.fen)
            finished, result = self.game_over(node.fen)

            if finished:  # recursion to the top, update values based on result
                # print('End')
                terminal_node_value = node.board_value * node.white_to_move
                node = self.update_values(node, terminal_node_value)  # todo: recursion


            elif not node.children:  # Spawn a child and recursion to the top
                # print('No children')
                node.addChildren(node, enumDict)
                terminal_node_value = node.board_value * node.white_to_move
                node = self.update_values(node, terminal_node_value)

            else:  # pick a child and go deeper
                # print('Explore')
                node = self.next_step(node)

        terminal_node_value = node.board_value * node.white_to_move
        node = self.update_values(node, terminal_node_value)
        # print('BestMove:', len(node.children), node.move, node.fen)
        # kala2 = [child.adjust_value for child in node.children]
        # print(kala2)

        print(node.fen)
        for child in node.children:
            
            print(child)
        
        print()

        move = self.returnBestMove(node.children)
        value = node.mean_value

        file_name = 'node_' + str(i) + '.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(node, f)

        return move, value

    def game_over(self, fen):  # place all your conditions here
        board = chess.Board(fen)
        finished = board.is_game_over()
        result = board.result()
        return finished, result

    def next_step(self, node):
        best_value = -np.inf
        for child in node.children:
            if child.adjust_value > best_value:  # should be probability based
                best_value = child.adjust_value
                best_node = child
        return best_node 

        # probs = [child.adjust_value+1 for child in node.children]
        # probs = np.array(probs)
        # probs /= probs.sum()
        # node = np.random.choice(node.children, p=probs)
        # best_node = node
        # return best_node 

    def returnBestMove(self, children):
        best_value = -np.inf
        for child in children:
            if child.adjust_value > best_value:  # should be probability based
                best_value = child.adjust_value
                best_move = child.move     
        return best_move       

    def update_values(self, node, terminal_node_value):  # recursion
        if node.parent:
            node.calcValue(terminal_node_value)
            parent = node.parent
            # parent.board_value = node.board_value #* node.white_to_move TODO!!!
            # parent.terminal_node_value = node.board_value
            parent.calcValue(terminal_node_value)
            return(self.update_values(parent, terminal_node_value))
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
