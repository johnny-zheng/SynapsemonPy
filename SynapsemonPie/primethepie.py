from __future__ import division
from pybrain.tools.shortcuts import buildNetwork                                
from pybrain.datasets import SupervisedDataSet                                  
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml.networkwriter import NetworkWriter 
from pybrain.tools.customxml.networkreader import NetworkReader
import numpy as np 

net = NetworkReader.readFrom('synapsemon.xml') 
# net = buildNetwork(96, 40, 1)

while True: 

	#build dataset
	ds = SupervisedDataSet(96, 1)
	for i in range(1000):
		inpt = np.random.randint(10, size=96)
		bar = np.random.randint(9) + 1
		for j in range(32):
			if inpt[j] < bar:
				inpt[j] = 0 
			else:
				inpt[j] = 1
		bar = np.random.randint(9) + 1
		for j in range(32, 64):
			if inpt[j] < bar:
				inpt[j] = 0 
			else:
				inpt[j] = 1
		for j in range(64, 96):
			inpt[j] = 0
		denom = sum(inpt)
		if denom == 0:
			denom = 1
		outpt = sum(inpt[:32]) / denom
		ds.addSample(inpt, outpt)

	#train network
	trainer = BackpropTrainer(net, ds)
	trainer.train()

	#save network
	NetworkWriter.writeToFile(net, 'synapsemon.xml')
	NetworkWriter.writeToFile(net, 'synapsemon_copy.xml')

# test the network
# for i in range(10):
# 	a = np.random.randint(2, size=96)
# 	for j in range(64, 96):
#  			a[j] = 0
# 	outpt = sum(a[:32]) / sum(a)
# 	print(outpt)
# 	print(net.activate(a))






