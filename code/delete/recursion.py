class Node:
    """
    Kala
    """

    def __init__(self, state, value, parent=None, children=None, visits=0):
        self.state = state
        self.parent = parent
        self.children = children
        self.value = value
        self.visits = visits


a1 = Node(1, 10)
a2 = Node(2, 21, a1)
a3 = Node(3, 31, a1)
a4 = Node(4, 41, a3)
a5 = Node(5, 51, a3)
a6 = Node(6, 61, a4)
a7 = Node(7, 71, a4)


def visitor(node):
    if node.parent:
        kala = node.parent
        print(kala.value)
        kala.value = kala.value + node.value
        return(visitor(kala))
    else:
        return 1


visitor(a7)

# for i in range(20):

import anxilliaryfunctions as af
import numpy as np

fen = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'
a = af.fenToTensor(fen)

sum_pieces = np.sum(a, axis=(1, 2))

# {'P':1,'N':2,'B':3,'R':4,'Q':5,'K':6,'p':7,'n':8,'b':9,'r':10,'q':11,'k':12}
piece_values = [1, 3, 3.5, 5, 9, 10, -1, -3, -3.5, -5, -9, -10]

total_value = sum_pieces * piece_values
bala = sum(total_value)
