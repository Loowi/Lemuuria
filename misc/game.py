#https://python-chess.readthedocs.io/en/v0.22.2/pgn.html

import chess
import chess.pgn

game = chess.pgn.Game()
game.headers["Event"] = "Example"
node = game.add_variation(chess.Move.from_uci("e2e4"))
node = node.add_variation(chess.Move.from_uci("e7e5"))
node = node.add_variation(chess.Move.from_uci("f2f4"))
game.headers["Result"] = '1'

print(game)


with open("loowi313.pgn", 'w') as new_pgn:
    exporter = chess.pgn.FileExporter(new_pgn)
    game.accept(exporter)
    

pgn = open("loowi313.pgn")

first_game = chess.pgn.read_game(pgn)

# Iterate through all moves and play them on a board.
board = first_game.board()
for move in first_game.main_line():
    board.push(move)

board
#Board('4r3/6P1/2p2P1k/1p6/pP2p1R1/P1B5/2P2K2/3r4 b - - 0 45')
