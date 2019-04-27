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
from SVM.svm import SVM

def plot_converge(ob_fct, c):

	filename = 'convergence_' + str(c) + '.png'

	conv = pd.DataFrame([ob_fct]).T 
	plot = conv.plot(linewidth = 3, figsize = (8,6), color = ['lightblue'], grid=True)

	plot.set_xlabel('updates', fontsize=12)
	plot.set_ylabel('Objective Function', fontsize=12)
	plot.set_title('Convergence with C = ' + str(c), fontsize=16)
	
	plt.savefig(filename)
	plt.show()

if __name__ == '__main__':

	svm_form = 'primal'
	r = [1]
	d = 0
	c = list()
	c_primal = [1/873, 10/873, 50/873, 100/873, 300/873, 500/873, 700/873]
	c_dual = [100/873, 500/873, 700/873]
	r_dual = [0.01, 0.1, 0.5, 1, 2, 5, 10, 100]
	num_epoch = 100

	parser = argparse.ArgumentParser()
	parser.add_argument("-s", "--svm", help="Choose the form of SVM to perform {primal, dual} [DEFAULT = primal]")
	parser.add_argument("-r", "--learn_rate", nargs='+', type=float, help="Choose the learning rate [DEFAULT] = 1 or from assignment for dual form")
	parser.add_argument("-d", "--learn_rate_tweak", help="Choose the learning rate tweak \"d\" [DEFAULT] = 0")
	parser.add_argument("-c", "--hyper_param", nargs='+', type=float, help="Choose the hyperparameter [DEFAULT] = default from assignment listing")
	parser.add_argument("-e", "--num_epoch", help="Choose the number of epochs for training [DEFAULT] = 100")
	parser.add_argument("-k", "--kernel", action='store_true', help="Flag to implement Gaussian kernel in the dual learning optimization")
	args = parser.parse_args()

	train_filename = 'datasets/bank-note/train.csv'
	test_filename = 'datasets/bank-note/test.csv'
	
	if args.svm:
		if args.svm != 'primal' and args.svm != 'dual':
			print 'invalid SVM form given; must be \'primal\' or \'dual\''
			exit()
		else:
			svm_form = args.svm
			
	if svm_form == 'primal':
		c = c_primal
	else:
		c = c_dual
		r = r_dual

	if args.learn_rate:
		r = args.learn_rate
	if args.learn_rate_tweak:
		d = float(args.learn_rate_tweak)
	if args.hyper_param:
		c = args.hyper_param
	if args.num_epoch:
		num_epoch = int(args.num_epoch)

	print 'Dataset file is "', train_filename
	print 'SVM form is "', svm_form
	print 'Learning rate "', r
	print 'hyperparameter "', c
	print 'Number of Epochs "', num_epoch
	print 'Kernel flag "', args.kernel

	train = list()
	test = list()
	with open(train_filename , 'r ') as f : 
		for line in f :
			terms = line.strip().split(',')
			float_terms = []
			for term in terms:
				float_terms.append(float(term))
			if float_terms[-1] == 0:
				float_terms[-1] = -1
			train.append(float_terms)

	with open(test_filename , 'r ') as f : 
		for line in f :
			terms = line.strip().split(',')
			float_terms = []
			for term in terms:
				float_terms.append(float(term))
			if float_terms[-1] == 0:
				float_terms[-1] = -1
			test.append(float_terms)

	train_data = np.array(train)
	test_data = np.array(test)

	print train_data	
	curr_alpha = list()
	print 'Performing SVM'
	for r_i in r:
		for c_i in c:
			svm = SVM(r_i, d, c_i, num_epoch)
			if svm_form == 'primal':
				weights, j = svm.primal(train_data)

				train_acc = svm.test_accuracy(train_data, weights)
				test_acc = svm.test_accuracy(test_data, weights)

				print "weight_vec = "
				print weights
				print "C setting = " + str(c_i)
				print "Training set accuracy = " + str(train_acc)
				print "Testing set accuracy = " + str(test_acc)
				print "Learning rate tweak = " + str(d)

				plot_converge(j, c_i)
			else:
				alpha = svm.dual_minimization(train_data, args.kernel)
				overlap_sup_vec = 0
				if len(curr_alpha) != 0:
					for i in range(len(curr_alpha)):
						if curr_alpha[i] == alpha[i]:
							overlap_sup_vec = overlap_sup_vec + 1

				x = list()
				y = list()

				for x_i in train_data:
					x.append(x_i[:-1])
					y.append(x_i[-1])

				# weights = np.zeros(len(x[0]))
				# num_sup_vec = 0
				# for i in range(len(x)):
				# 	if alpha[i] != 0:
				# 		num_sup_vec = num_sup_vec + 1
				# 	weights = weights + (alpha[i]*y[i]*x[i])
				# weights = sum(alpha[i]*y[i]*x[i] for i in range(len(x)))

				# print "Number of support vectors = " + str(num_sup_vec)
				# print "Number of overlapped support vectors with previous gamma = " + str(overlap_sup_vec)
				# print "weight_vec = "
				# print weights
				print "C setting = " + str(c_i)
				print "Learning rate = " + str(r_i)
				# curr_alpha = alpha
				train_acc = svm.test_dual_acc(train_data, alpha, train_data, args.kernel)
				test_acc = svm.test_dual_acc(test_data, alpha, train_data, args.kernel)
				print "Training set accuracy = " + str(train_acc)
				print "Testing set accuracy = " + str(test_acc)
				

	