import glob
import chess
import logging
from multiprocessing import Pool
from os import getpid
import chess.pgn
import pandas as pd
import numpy as np

logger = logging.getLogger('Main')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(f"%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                              datefmt=f"%m/%d/%Y %H:%M:%S")

ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)


def process(file):
    logger = logging.getLogger('Main.ProcessFiles')
    logger.info('Started process: %s', getpid())

    pgn = open(file)
    games = []
    i = 0
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        games.append(game)
        i = i+1
        if i % 10000 == 0:
            logger.info('Process %s processed %s games', getpid(), i)

    logger.info('Process %s collected %s games', getpid(), len(games))

    positions = []
    for gameNum, game in enumerate(games):
        board = game.board()
        winlose = game.headers["Result"]
        outcomes = {'0-1': -1, '1-0': 1, '1/2-1/2': 0}
        win = outcomes[winlose]

        gameID = game.headers["FICSGamesDBGameNo"]
        if gameNum % 10000 == 0:
            logger.info(
                'Process %s has moves for %s games', getpid(), gameNum)

        for moveNum, move in enumerate(game.mainline_moves()):
            board.push(move)
            fen = board.fen()

            # Store the result
            positions.append([gameID, np.uint32(gameNum),
                              np.uint32(moveNum), fen, np.int8(win), str(move)])

    df = pd.DataFrame(positions, columns=[
                      'ID', 'Game', 'MoveNum', 'Fen', 'Result', 'Move'])
    df['Move'] = df['Move'].shift(-1)
    df['ID'] = pd.to_numeric(df['ID'], downcast='unsigned')

    # Save the file
    result = file.split('\\')[-1].split('.')[0]
    fileName = ".".join([result, 'pkl'])
    df.to_pickle(fileName)

    logger.info('Process %s has finished', getpid())

    return 1


if __name__ == '__main__':
    logger.info('Started the process')
    inputFolder = 'C:\\Users\\Watson\\Projects\\remus\\input\\games\\*.pgn'
    files = glob.glob(inputFolder)

    logger.info('Number of files to be processed: %s', len(files))

    pool = Pool(3)
    pool.map(process, files)
    pool.close()

    logger.info('Finished pickling the results')
