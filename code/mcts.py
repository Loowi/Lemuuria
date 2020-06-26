from collections import defaultdict
import random
import chess
from game import Game
import anxilliaryfunctions as af
from pathlib import WindowsPath
from tensorflow import keras
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

        gameState = af.fenToTensor(self.fen)
        inputState = np.expand_dims(gameState, axis=0)
        inputState = inputState.astype(np.float32)
        
        input_name = self.session.get_inputs()[0].name
        output_name_1 = self.session.get_outputs()[0].name
        output_name_2 = self.session.get_outputs()[1].name
        
        policy, value = self.session.run([output_name_1, output_name_2], {input_name: inputState})

        # policy, value = self.model.predict(inputState)

        finalDict, enumDict = af.createMoveDict()
        values = {move: policy[0][enumDict[move].value] for move in moves}

        self.board_value = value

        for move in values.keys():
            boardTemp = chess.Board(self.fen)
            boardTemp.push(chess.Move.from_uci(move))
            prior_probability = values[move]
            node = MctsTree(self.session, boardTemp, prior_probability, move)
            self.children.append(node)

        kala2 = 5

        # moves = [str(x) for x in list(game.board.legal_moves)]

        # kala = {move: policy[0][enumDict[move].value] for move in moves}
        # move = max(kala, key=kala.get)

        # objects = [MctsNode(move, value)
        #            for move, value in zip(moves, self.calc_policy(moves))]

        # gameState = af.fenToTensor(self.state)
        # inputState = np.expand_dims(gameState, axis=0)
        # policy, value = model.predict(inputState)

    # def addChildren(self, moves, model):
    #     gameState = af.fenToTensor(self.state)
    #     inputState = np.expand_dims(gameState, axis=0)
    #     policy, value = model.predict(inputState)

    #     # Extract move with the highest value: find legit moves, max index and lookup in the dict
    #     moves = [str(x) for x in list(game.board.legal_moves)]
    #     kala = {move: policy[0][enumDict[move].value] for move in moves}
    #     move = max(kala, key=kala.get)

    #     objects = [MctsNode(move, value)
    #                for move, value in zip(moves, self.calc_policy(moves))]
    #     self.children = self.children + objects

    # def pickBest(self):
    #     return self.children[0]

    # def update(self):
    #     self.num_visits += 1

    # def calc_policy(self, moves):
    #     policyValues = random.sample(range(10, 30), len(moves))
    #     return policyValues

    # def nextMove(self):
    #     pass


# game = chess.Board()
# state = game.board_fen()
# node = MctsNode(state, 0)


# i = 0

# while i < 10:
#     i += 1
#     node.update()

#     gameState = game.board.fen()
#     gameState = af.fenToTensor(gameState)

#     inputState = np.expand_dims(gameState, axis=0)
#     policy, value = model.predict(inputState)
#     # moves = [str(x) for x in list(game.board.legal_moves)]

#     if not node.children:
#         possible_moves = [str(x) for x in list(game.board.legal_moves)]
#         node.addChildren(possible_moves, model)

#     # node = MctsNode(state, 0)
#     # best = node.pickBest()
#     # makemove with chess


# ba = node.children

# for object in ba:
#     print(object.mean_action_value)


# find index
# values.index(min(values))


#  # BACKUP STEP
#         # on returning search path
#         # update: N, W, Q
#         with self.node_lock[state]:
#             my_visit_stats.sum_n += -virtual_loss + 1
#             my_stats.n += -virtual_loss + 1
#             my_stats.w += virtual_loss + leaf_v
#             my_stats.q = my_stats.w / my_stats.n

#         return leaf_v

# python -m cProfile -s tottime main.py

#  my_visitstats = self.tree[state]

#         if my_visitstats.p is not None: #push p to edges
#             tot_p = 1e-8
#             for mov in env.board.legal_moves:
#                 mov_p = my_visitstats.p[self.move_lookup[mov]]
#                 my_visitstats.a[mov].p = mov_p
#                 tot_p += mov_p
#             for a_s in my_visitstats.a.values():
#                 a_s.p /= tot_p
#             my_visitstats.p = None

#         xx_ = np.sqrt(my_visitstats.sum_n + 1)  # sqrt of sum(N(s, b); for all b)

#         e = self.play_config.noise_eps
#         c_puct = self.play_config.c_puct
#         dir_alpha = self.play_config.dirichlet_alpha

#         best_s = -999
#         best_a = None
#         if is_root_node:
#             noise = np.random.dirichlet([dir_alpha] * len(my_visitstats.a))

#         i = 0
#         for action, a_s in my_visitstats.a.items():
#             p_ = a_s.p
#             if is_root_node:
#                 p_ = (1-e) * p_ + e * noise[i]
#                 i += 1
#             b = a_s.q + c_puct * p_ * xx_ / (1 + a_s.n)
#             if b > best_s:
#                 best_s = b
#                 best_a = action

#         return best_a
