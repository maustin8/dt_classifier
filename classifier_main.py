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
from DecisionTree.dt import decision_tree
from EnsembleLearning.ensemble import AdaBoost
from EnsembleLearning.ensemble import BaggedTrees
from EnsembleLearning.ensemble import RandomForest

def splitdataset(balance_data): 
  
	# Seperating the target variable 
	X = balance_data.values[:, 1:5] 
	Y = balance_data.values[:, 0] 
  
	# Spliting the dataset into train and test 
	X_train, X_test, y_train, y_test = train_test_split(  
	X, Y, test_size = 0.3, random_state = 100) 
      
	return X, Y, X_train, X_test, y_train, y_test 
def test(test_data, tree):
	
	index = 0
	num_wrong = 0
	num_correct = 0
	while index < len(test_data.index):
		instance = test_data.iloc[ index , : ]
		predict_result = tree.predict(instance)
		test_result = test_data.iloc[ index , : ][test_data.keys()[-1]]
		if predict_result != test_result:
			num_wrong += 1
		else:
			num_correct += 1

		index += 1
	return num_correct, num_wrong

def get_error_rate(prediction, label):
		return sum(prediction != label) / float(len(label))

def plot_error_rate(train_error, test_error, ensemble_type, dataset, sample_size=0):

	filename = dataset + '_'
	error = pd.DataFrame([train_error, test_error]).T 
	error.columns = ['Training', 'Test']
	plot = error.plot(linewidth = 3, figsize = (8,6), color = ['lightblue','darkblue'], grid=True)
	filename += ensemble_type + '_' + str(len(train_error))
	if ensemble_type == 'adaboost':
		plot.set_xlabel('Num of iterations', fontsize=12)
	else:
		plot.set_xlabel('Num of trees', fontsize=12)
	# plot.set_xticklabels(range(0,450,50))
	plot.set_ylabel('Error rate', fontsize=12)
	if ensemble_type == 'adaboost':
		plot.set_title(ensemble_type + ' Training and Testing errors vs Number of Trees', fontsize=16)
	elif ensemble_type == 'rand_forest':
		filename += '_' + str(sample_size)
		plot.set_title(ensemble_type + ' ' + str(sample_size) + ': ' + ' Training and Testing errors vs Number of Trees', fontsize=16)
	else:
		plot.set_title(ensemble_type + ' Training and Testing errors vs Number of Trees', fontsize=16)
	plt.axhline(y=test_error[0], linewidth=1, color = 'red', ls = 'dashed')
	
	plt.savefig(filename)
	# plt.show()



if __name__ == '__main__':

	dataset_file = ''
	gain_type = 'entropy'
	ensemble_type = 'tree'
	num_iter = 1000
	num_trees = 1000
	sample_size = 0
	feature_sample_size = 0
	max_tree_depth = str(inf)

	car_attributes = ['buying','maint','doors','persons','lug_boot','safety','label']

	car_data = {'buying': 'cat','maint': 'cat','doors': 'cat','persons': 'cat','lug_boot': 'cat', \
	'safety': 'cat','label': 'cat'}

	tennis_attributes = ['outlook', 'temperature', 'humidity', 'wind', 'label']
	tennis_data = {'outlook': 'cat','temperature': 'cat','humidity': 'cat','wind': 'cat','label': 'cat'}

	bank_attributes = ['age', 'job', 'marital','education', 'default', 'balance', 'housing', \
	'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']

	bank_data = {'age': 'num', 'job': 'cat', 'marital': 'cat', 'education': 'cat', 'default': 'cat', \
	'balance': 'num', 'housing': 'cat', 'loan': 'cat', 'contact': 'cat', 'day': 'num', 'month': 'cat', \
	'duration': 'num', 'campaign': 'num', 'pdays': 'num', 'previous': 'num', 'poutcome': 'cat', 'label':'cat'}	

	concrete_attributes = ['cement', 'slag', 'fly_ash', 'water', 'sp', 'coarse_aggr', 'fine_aggr', 'output']

	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", help="Choose between the datasets in datasets/ subfolder {car, bank, tennis}")
	parser.add_argument("-e", "--ensemble_type", help="Choose the type of ensemble learning {tree, adaboost, bagged, rand_forest} [DEFAULT = tree]")
	parser.add_argument("-t", "--num_iterations", help="Choose number of iterations [DEFAULT = 1000]")
	parser.add_argument("-n", "--num_trees", help="Choose number of trees for bagged methods [DEFAULT = 1000]")
	parser.add_argument("-s", "--sample_size", help="Choose sample size for bagged methods [DEFAULT = 2]")
	parser.add_argument("-g", "--gain", help="Choose method of choosing best split {entropy, gini, majority_error} [DEFAULT = entropy]")
	parser.add_argument("-d", "--tree_depth", help="Specify maximum tree depth")
	parser.add_argument("-f", "--feature_sample_size", help="Specify feature sample size for random forest")
	parser.add_argument("-u", "--unknown", action='store_true', help="Flag to remplace unknown values with most common one")
	args = parser.parse_args()

	train_filename = ''
	test_filename = ''
	attribute_list = dict()
	attributes = []
	data_types = dict()
	
	if args.dataset != 'car' and args.dataset != 'bank' and args.dataset != 'tennis':
		print 'invalid dataset given for tree method; must be \'car\' or \'bank\' or \'tennis\''
		exit()
	elif args.dataset == 'car':
		attributes = car_attributes
		attribute_list = car_data
		train_filename = 'datasets/car/train.csv'
		test_filename = 'datasets/car/test.csv'
	elif args.dataset == 'bank':
		attributes = bank_attributes
		attribute_list = bank_data
		train_filename = 'datasets/bank/train.csv'
		test_filename = 'datasets/bank/test.csv'
	elif args.dataset == 'tennis':
		attributes = tennis_attributes
		attribute_list = tennis_data
		train_filename = 'datasets/tennis/train2.csv'

	train_dataset_file = train_filename
	test_dataset_file = test_filename
	
	if args.ensemble_type:
		if args.ensemble_type != 'tree' and args.ensemble_type != 'adaboost' and args.ensemble_type != 'bagged' and args.ensemble_type != 'rand_forest':
			print 'invalid ensemble learning type given; must be \'tree\' or \'adaboost\' or \'bagged\' or \'rand_forest\''
			exit()
		else:
			ensemble_type = args.ensemble_type
	if args.num_iterations:
		num_iter = args.num_iterations
	if args.num_trees:
		num_trees = args.num_trees
	if args.sample_size:
		sample_size = args.sample_size
	if args.feature_sample_size:
		feature_sample_size = args.feature_sample_size
	if args.gain:
		if args.gain != 'entropy' and args.gain != 'gini' and args.gain != 'majority_error':
			print 'invalid gain given; must be \'entropy\' or \'gini\' or \'majority_error\''
			exit()
		else:
			gain_type = args.gain
	if args.tree_depth:
		if float(args.tree_depth) <= 0:
			print 'tree_depth must be greater than 0'
			exit()
		else:
			max_tree_depth = args.tree_depth
	print 'Dataset file is "', train_dataset_file
	print 'Ensemble learning type is "', ensemble_type
	if args.unknown:
		print 'Replacing unkowns with most common value'


	train_file = np.genfromtxt(train_dataset_file, delimiter=',', names=attributes, dtype=None)
	train_data = pd.DataFrame(train_file, columns=attributes)
	
	test_file = np.genfromtxt(test_dataset_file, delimiter=',', names=attributes, dtype=None)
	test_data = pd.DataFrame(test_file, columns=attributes)

	for attribute, data_type in attribute_list.iteritems():
			if data_type == 'num':

				median_train = train_data[attribute].median()
				median_test = test_data[attribute].median()
				for index, row in train_data.iterrows():
					if float(row[attribute]) >= median_train:
						train_data.iloc[index, train_data.columns.get_loc(attribute)] = '+'
					else:
						train_data.iloc[index, train_data.columns.get_loc(attribute)] = '-'

				for index, row in test_data.iterrows():
					if float(row[attribute]) >= median_train:
						test_data.iloc[index, test_data.columns.get_loc(attribute)] = '+'
					else:
						test_data.iloc[index, test_data.columns.get_loc(attribute)] = '-'
			values_train = np.unique(train_data[attribute])
			values_test = np.unique(test_data[attribute])
			# replace unknown values if -u flag was used
			if args.unknown:
				if 'unknown' in values_train:
					max_value = ''
					max_idx = 0
					for value in values_train:
						count = train_data[attribute].value_counts()[value]
						if max_idx < count:
							max_idx = count
							max_value = value
					train_data[attribute].replace('unknown', max_value, inplace=True)

				if 'unknown' in values_test:
					max_value = ''
					max_idx = 0
					for value in values_test:
						count = test_data[attribute].value_counts()[value]
						if max_idx < count:
							max_idx = count
							max_value = value
					test_data[attribute].replace('unknown', max_value, inplace=True)
	
	if not args.sample_size:
		sample_size = train_data.shape[0]

	if not args.feature_sample_size:
		feature_sample_size = len(attributes)

	attributes = []
	attribute_values = dict()
	for attribute, data_type in attribute_list.iteritems(): 
		attributes.append(attribute)
		attribute_values[attribute] = np.unique(test_data[attribute])
	attributes.remove('label')

	if ensemble_type == 'tree':
		print 'Gain type is "', gain_type
		print 'Max tree depth is "', max_tree_depth
		# train decision tree			
		print 'started making tree'
		tree = decision_tree(train_data, attributes, attribute_values, gain_type, max_tree_depth, args.unknown)
		print 'finished tree'
		# pprint.pprint(tree.get_tree())
		print 'started making tree test'
		tree2 = dt(train_filename, test_filename, attributes, attribute_values, gain_type, max_tree_depth, args.unknown)
		print 'finished tree test'
		# test decision tree with training data and testing data
		# correct_train, wrong_train = test(train_data, tree)
		# percentage_train = float(correct_train) / (correct_train + wrong_train) * 100
		# print 'TRAINING DATASET'
		# print 'number correct = ' + str(correct_train)
		# print 'number wrong = ' + str(wrong_train)
		# print 'accuracy = ' + str(percentage_train)

		# test_data = tree.massage_data(test_data, attribute_list, args.unknown)
		correct_test, wrong_test = test(test_data, tree)
		percentage_test = float(correct_test) / (correct_test + wrong_test) * 100
		
		print 'TESTING DATASET'
		print 'number correct = ' + str(correct_test)
		print 'number wrong = ' + str(wrong_test)
		print 'accuracy = ' + str(percentage_test)

	elif ensemble_type == 'adaboost':
		print 'Number of iterations "', num_iter
		train_error = []
		test_error = []
		i = 1
		while i <= int(num_iter):
			print 'num_iter = ' + str(i)
			boost = AdaBoost(train_data, test_data, attributes, attribute_values, i)
			pred_train, pred_test = boost.get_predictions()
			train_error.append(get_error_rate(pred_train, train_data.label))
			test_error.append(get_error_rate(pred_test, test_data.label)) 
			i += 1
		plot_error_rate(train_error, test_error, ensemble_type, args.dataset)
		# print plot of train and test errors vs testing iterations
		# print plot of train and test errors of decision stumps learned in each iteration

	elif ensemble_type == 'bagged':
		print 'Number of trees "', num_trees
		print 'Sample size "', sample_size
		train_error = []
		test_error = []
		i = 1
		while i <= int(num_trees):
			print 'current number of trees ' + str(i)
			bag = BaggedTrees(train_data, test_data, attributes, attribute_values, i, sample_size)
			pred_train, pred_test = bag.get_predictions()
			train_error.append(get_error_rate(pred_train, train_data.label))
			test_error.append(get_error_rate(pred_test, test_data.label)) 
			i += 1
		plot_error_rate(train_error, test_error, ensemble_type, args.dataset)
		# print plot of train and test errors vs number of trees
		# compare against single trees

	elif ensemble_type == 'rand_forest':
		print 'Number of trees "', num_trees
		print 'Sample size "', sample_size
		print 'Feature subset size ' + str(feature_sample_size)
		train_error = []
		test_error = []
		i = 1
		while i <= int(num_trees):
			print 'current number of trees ' + str(i)
			forest = RandomForest(train_data, test_data, attributes, attribute_values, i, sample_size, feature_sample_size)
			pred_train, pred_test = forest.get_predictions()
			train_error.append(get_error_rate(pred_train, train_data.label))
			test_error.append(get_error_rate(pred_test, test_data.label)) 
			i += 1
		plot_error_rate(train_error, test_error, ensemble_type, args.dataset, feature_sample_size)
		# print plot of train and test errors vs number of trees for each feature subset size setting
	
	

