#Pretraining the weights
from __future__ import division
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.datasets import SupervisedDataSet                                  
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
import random

# Constants
BLACK, WHITE = 0, 1
#neural net 
#20, 50, 80, 110, 140
net = buildNetwork(97, 140, 1)

def move_function(board):
    global net
    #active player
    #if board.active == BLACK:
    #   ds.addSample(board_to_input(board), max_value)
    #    best_move = best_max_move
    #elif board.active == WHITE: 
    #    ds.addSample(board_to_input(board), min_value)
    #   best_move = best_min_move
    boardString=board_to_input(board)
    black_material=0
    white_material=0
    for i in range(32):
        isKing=boardString[i+64]=='1'
        if boardString[i]=='1':
            if isKing:
                black_material=black_material+2
            else: 
                black_material=black_material+1
        if boardString[i+32]=='1':
            if isKing:
                white_material=white_material+2
            else:
                white_material=white_material+1
    
    board_val = black_material/(black_material+white_material)

    #create a new dataset. Add a sample with board as input, value as output
    ds = SupervisedDataSet(97, 1)
    ds.addSample(boardString, board_val)
    trainer = BackpropTrainer(net, ds)
    trainer.train()
    NetworkWriter.writeToFile(net, 'SynapsemonPie/synapsemon_primer1.xml')
    NetworkWriter.writeToFile(net, 'SynapsemonPie/synapsemon_primer1_copy.xml')

    return random.choice(board.get_moves()) 

def board_to_input(board):
    EMPTY = -1
    BLACK_KING = 2
    WHITE_KING = 3

    if board.active == BLACK:
        black_kings = board.backwa rd[board.active]
        black_men = board.forward[board.active] ^ black_kings
        white_kings = board.forward[board.passive]
        white_men = board.backward[board.passive] ^ white_kings
    else:
        black_kings = board.backward[board.passive]
        black_men = board.forward[board.passive] ^ black_kings
        white_kings = board.forward[board.active]
        white_men = board.backward[board.active] ^ white_kings

    state = [[None for _ in range(8)] for _ in range(4)]
    for i in range(4):
        for j in range(8):
            cell = 1 << (9*i + j)
            if cell & black_men:
                state[i][j] = BLACK
            elif cell & white_men:
                state[i][j] = WHITE
            elif cell & black_kings:
                state[i][j] = BLACK_KING
            elif cell & white_kings:
                state[i][j] = WHITE_KING
            else:
                state[i][j] = EMPTY

    #flatten list 
    state = [item for sublist in state for item in sublist]

    inpt = [0] * 97

    for i in range(32):
        if state[i] == BLACK or state[i] == BLACK_KING:
            inpt[i] = 1
        elif state[i] == WHITE or state[i] == WHITE_KING:
            inpt[i + 32] = 1
        if state[i] == BLACK_KING or state[i] == WHITE_KING:
            inpt[i + 64] = 1

    if board.active = BLACK:
        inpt[96] = 1
    else:
        inpt[96] = 0

    return inpt 