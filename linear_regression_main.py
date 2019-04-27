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
from LinearRegression.linear_regress import gradient

def plot_cost_fct(cost_fct, descent_type, r):

	filename = descent_type + '_cost_fct'
	cost = pd.DataFrame([cost_fct]).T 
	cost.columns = ['Cost Function']
	plot = cost.plot(linewidth = 3, figsize = (8,6), color = ['darkblue'], grid=True)
	
	plot.set_xlabel('Num of steps', fontsize=12)
	
	# plot.set_xticklabels(range(0,450,50))
	plot.set_ylabel('Cost Function', fontsize=12)
	
	plot.set_title('Cost Function vs Steps with learning rate: ' + str(r), fontsize=16)
	
	plt.savefig(filename)
	plt.show()

if __name__ == '__main__':
	dataset_file = ''
	descent_type = 'batch'
	r = 0.1

	concrete_attributes = ['cement', 'slag', 'fly_ash', 'water', 'sp', 'coarse_aggr', 'fine_aggr', 'output']
	test_attributes = ['whatever','whatever2','whatever3','output']

	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--descent", help="Choose method linear regression descent {batch, stochastic} [DEFAULT = batch]")
	parser.add_argument("-r", "--learn_rate", help="Choose the learning rate [DEFAULT] = 1")
	parser.add_argument("-n", "--num_iter", help="Choose the number of iterations for the gradient algorithms [DEFAULT = size of data")
	args = parser.parse_args()

	train_filename = 'datasets/concrete/train.csv'
	test_filename = 'datasets/concrete/test.csv'

	# train_filename = 'datasets/test/train.csv'
	
	if args.descent:
		if args.descent != 'batch' and args.descent != 'stochastic':
			print 'invalid descent given; must be \'batch\' or \'stochastic\''
			exit()
		else:
			descent_type = args.descent

	if args.learn_rate:
		r = args.learn_rate

	print 'Dataset file is "', train_filename
	print 'descent type is "', descent_type
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

	test_data = np.array(test)

	num_iter = train_data.shape[0]

	if args.num_iter:
		num_iter = float(args.num_iter)

	print 'num_iter = ' + str(num_iter)

	Y = list()
	X = list()
	for example in train_data:
		Y.append(example[-1])
		X.append(example[:-1])
	xT = np.transpose(X)
	xTx = np.matmul(xT, X)
	inv = np.linalg.inv(xTx)
	invxT = np.matmul(inv, xT)
	opt_weight = np.matmul(invxT, Y)
	
	print 'Optimal weight Calculated Analytically: '
	print opt_weight
	
	print 'Performing Gradient descent'
	grad = gradient(train_data, r, num_iter, grad_type=descent_type)
	weight_vec, bias, cost_fct = grad.get_results()
	print 'Final weight vector = '
	print weight_vec
	print 'Final bias = '
	print bias
	test_cost = grad.calc_cost(test_data, weight_vec, bias)
	print 'Cost Function Value of test data = ' + str(test_cost)
	plot_cost_fct(cost_fct, descent_type, r)
		# print plot of cost function with each training step
		# use final weight vector to calculate cost function of test data
		# print plot of cost function with each training step
		# use final weight vector to calculate cost function of test data

	