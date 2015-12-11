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
net = NetworkReader.readFrom('SynapsemonPie/synapsemon_primer50.xml') 

def boardVal(boardList):
    black_material=0
    white_material=0
    for i in range(32):
        isKing=boardList[i+64]== 1
        if boardList[i]== 1:
            if isKing:
                black_material=black_material+2
            else: 
                black_material=black_material+1
        if boardList[i+32]== 1:
            if isKing:
                white_material=white_material+2
            else:
                white_material=white_material+1
    
    board_val = black_material/(black_material+white_material)
    return board_val

def makeBoard():
    boardList=[0] * 97
    for i in range(32):
        piece=(random.randint(1,2)==1)
        black=(random.randint(1,2)==1)
        king=(random.randint(1,12)==1)
        if piece:
            if king:
                boardList[i+64]=1
            if black:
                boardList[i]=1
            else:
                boardList[i+32]=1
    boardList[96]=random.randint(0,1)
    return boardList

def storeBoards():
    ds = SupervisedDataSet(97,1)
    for i in range(1000):
        boardList=makeBoard()
        ds.addSample(boardList, boardVal(boardList))
    ds.saveToFile('SynapsemonPie/boards')

def testNets():
    ds = SupervisedDataSet.loadFromFile('SynapsemonPie/boards')
    net20 = NetworkReader.readFrom('SynapsemonPie/synapsemon_primer20.xml') 
    net50 = NetworkReader.readFrom('SynapsemonPie/synapsemon_primer50.xml') 
    net80 = NetworkReader.readFrom('SynapsemonPie/synapsemon_primer80.xml') 
    net110 = NetworkReader.readFrom('SynapsemonPie/synapsemon_primer110.xml') 
    net140 = NetworkReader.readFrom('SynapsemonPie/synapsemon_primer140.xml') 
    trainer20 = BackpropTrainer(net20, ds)
    trainer50 = BackpropTrainer(net50, ds)
    trainer80 = BackpropTrainer(net80, ds)
    trainer110 = BackpropTrainer(net110, ds)
    trainer140 = BackpropTrainer(net140, ds)
    print trainer20.train()
    print trainer50.train()
    print trainer80.train()
    print trainer110.train()
    print trainer140.train()

#storeBoards()
testNets()