from __future__ import division
import numpy as np 
import pandas as pd 
eps = np.finfo(float).eps
inf = np.finfo(float).max
import sys, getopt
import os
import argparse
import csv
import pprint
import collections
import matplotlib.pyplot as plt
from NeuralNetworks.nn import NN
from NeuralNetworks.bonus import tensor

def plot_converge(ob_fct, w):

	filename = 'convergence_' + str(w) + '.png'

	conv = pd.DataFrame([ob_fct]).T 
	plot = conv.plot(linewidth = 3, figsize = (8,6), color = ['lightblue'], grid=True)

	plot.set_xlabel('updates', fontsize=12)
	plot.set_ylabel('Objective Function', fontsize=12)
	plot.set_title('Convergence with width = ' + str(w), fontsize=16)
	
	plt.savefig(filename)
	# plt.show()

if __name__ == '__main__':

	r = 1
	d = 1
	width = [5, 10, 25, 50, 50]
	num_epoch = 100
	depth = [3, 5, 9]

	parser = argparse.ArgumentParser()
	parser.add_argument("-r", "--learn_rate", help="Choose the initial learning rate [DEFAULT] = 1")
	parser.add_argument("-d", "--learn_rate_tweak", help="Choose the learning rate tweak \"d\" [DEFAULT] = 1")
	parser.add_argument("-w", "--width", nargs='+', type=float, help="Choose the width of each hidden layer [DEFAULT] = default from assignment listing")
	parser.add_argument("-e", "--num_epoch", help="Choose the number of epochs for training [DEFAULT] = 100")
	parser.add_argument("-l", "--depth", nargs='+', type=float, help="Choose the depths of the neural network for bonus [DEFAULT] = default from assignment listing")
	parser.add_argument("-tf", "--tensor", action='store_true', help="Flag to use tensorflow library for neural network implementation")
	args = parser.parse_args()

	train_filename = 'datasets/bank-note/train.csv'
	test_filename = 'datasets/bank-note/test.csv'
	

	if args.learn_rate:
		r = args.learn_rate
	if args.learn_rate_tweak:
		d = float(args.learn_rate_tweak)
	if args.width:
		width = args.width
	if args.num_epoch:
		num_epoch = int(args.num_epoch)
	if args.depth:
		depth = args.depth

	print('Dataset file is "', train_filename)
	print('Learning rate "', r)
	print('Learning rate tweak "', d)
	print('width "', width)
	print('Number of Epochs "', num_epoch)
	print('TF NN depth "', depth)

	train = list()
	test = list()
	with open(train_filename) as f : 
		for line in f :
			terms = line.strip().split(',')
			float_terms = []
			for term in terms:
				float_terms.append(float(term))
			train.append(float_terms)

	with open(test_filename) as f : 
		for line in f :
			terms = line.strip().split(',')
			float_terms = []
			for term in terms:
				float_terms.append(float(term))
			test.append(float_terms)

	train_data = np.array(train)
	test_data = np.array(test)
	print('column number = ' + str(train_data.shape[1] - 1))
	# nn = NN(r, d, 3, num_epoch, train_data.shape[1] - 1)
	# print train_data	
	print('Performing Neural Network train and test')
	if args.tensor:
		activations = ['tanh', 'relu']
		# activations = ['relu']
		x_train = list()
		y_train = list()
		x_test = list()
		y_test = list()
		for x in train_data:
			x_train.append(x[:-1])
			y_train.append(x[-1])
		for x in test:
			x_test.append(x[:-1])
			y_test.append(x[-1])

		x_train = np.array(x_train)
		y_train = np.array(y_train)
		x_test = np.array(x_test)
		y_test = np.array(y_test)


		for activation in activations:
			for d in depth:
				for w in width:
					print("activation = {} depth = {} width = {}".format(activation, d, w))
					nn = tensor(x_train.shape, 2, w, d, num_epoch)
					nn.train_and_evaluate(activation, x_train, y_train, x_test, y_test)
	else:
		for w in width:
			print('width = ' + str(w))
			nn = NN(r, d, w, num_epoch, train_data.shape[1] - 1)
			# weights = nn.get_weights()
			net = nn.train(train_data)

			# train_acc = nn.testing_accuracy(train_data, net)
			# test_acc = nn.testing_accuracy(test_data, net)

			# print("Training set accuracy = " + str(train_acc))
			# print("Testing set accuracy = " + str(test_acc))
		
				

	