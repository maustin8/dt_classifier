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
from Perceptron.percept import Perceptron

if __name__ == '__main__':
	dataset_file = ''
	percept_type = 'standard'
	r = 1

	concrete_attributes = ['cement', 'slag', 'fly_ash', 'water', 'sp', 'coarse_aggr', 'fine_aggr', 'output']
	test_attributes = ['whatever','whatever2','whatever3','output']

	parser = argparse.ArgumentParser()
	parser.add_argument("-p", "--percept", help="Choose the type of perceptron to perform {standard, voted, average} [DEFAULT = standard]")
	parser.add_argument("-r", "--learn_rate", help="Choose the learning rate [DEFAULT] = 1")
	parser.add_argument("-e", "--num_epoch", help="Choose the number of epochs for training [DEFAULT = 10")
	args = parser.parse_args()

	train_filename = 'datasets/bank-note/train.csv'
	test_filename = 'datasets/bank-note/test.csv'
	
	if args.percept:
		if args.percept != 'standard' and args.percept != 'voted' and args.percept != 'average':
			print 'invalid descent given; must be \'standard\' or \'voted\' or \'average\''
			exit()
		else:
			percept_type = args.percept

	if args.learn_rate:
		r = float(args.learn_rate)

	print 'Dataset file is "', train_filename
	print 'perceptron type is "', percept_type
	print 'Learning rate "', r

	train = list()
	test = list()
	with open(train_filename , 'r ') as f : 
		for line in f :
			terms = line.strip().split(',')
			float_terms = []
			for term in terms:
				float_terms.append(float(term))
			train.append(float_terms)

	with open(test_filename , 'r ') as f : 
		for line in f :
			terms = line.strip().split(',')
			float_terms = []
			for term in terms:
				float_terms.append(float(term))
			test.append(float_terms)

	train_data = np.array(train)
	for example in train_data:
		if example[-1] == 0:
			example[-1] = -1

	test_data = np.array(test)
	for example in test_data:
		if example[-1] == 0:
			example[-1] = -1

	num_epoch = 10

	if args.num_epoch:
		num_epoch = float(args.num_epoch)

	print 'num_epoch = ' + str(num_epoch)

	print 'Performing Perceptron'
	percept = Perceptron(train_data, test_data, r, num_epoch, percept_type=percept_type)
	if percept_type == 'voted':
		weight_vec, cost, pred_acc = percept.get_voted_results()
		print 'Number of weight vectors:'
		print len(weight_vec)
		print 'Final weight vectors and their costs'
		for i in range(len(weight_vec)):
			print weight_vec[i]
			print cost[i]
		print 'Final prediction accuracy = '
		print pred_acc
	else:
		weight_vec, pred_acc = percept.get_results()
		print 'Final weight vector = '
		print weight_vec
		print 'Final prediction accuracy = '
		print pred_acc

	