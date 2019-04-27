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
import random
from random import randrange
from DecisionTree.dt import decision_tree

class AdaBoost():
	def __init__(self, train, test, attributes, attribute_values, num_iter):
		self.attributes = attributes
		self.attribute_values = attribute_values

		for index, row in train.iterrows():
			if row['label'] == 'no':
				train.iloc[index, train.columns.get_loc('label')] = '-1'
			else:
				train.iloc[index, train.columns.get_loc('label')] = '1'
		for index, row in test.iterrows():
			if row['label'] == 'no':
				test.iloc[index, test.columns.get_loc('label')] = '-1'
			else:
				test.iloc[index, test.columns.get_loc('label')] = '1'

		self.train_pred, self.test_pred = self.boost(train, test, self.attributes, num_iter)

	def boost(self, train, test, attributes, num_iter):
		m = train.shape[0]
		weights = np.ones(m) / m
		predict_train_iter, predict_test_iter = [np.zeros(m), np.zeros(test.shape[0])]
		for i in range(int(num_iter)):
			print(weights)
			print train.shape[0]
			stump = self.create_stump(train, weights)

			pprint.pprint(stump.get_tree())
			
			h_train = []
			h_test = []
			index = 0
			while index < len(train.index):
				instance = train.iloc[ index , : ]
				predict_result = stump.predict(instance)
				h_train.append(predict_result)
				index += 1
			index = 0
			# test = stump.massage_data(test, test_att_list)
			while index < len(test.index):
				instance = test.iloc[ index , : ]
				predict_result = stump.predict(instance)
				h_test.append(predict_result)
				index += 1
			y = train.label
			miss = [int(x) for x in (h_train != y)]
			yh = [x if x==1 else -1 for x in miss]
			w_error = np.dot(weights, miss) / sum(weights)
			alpha = 0.5 * np.log((1-w_error)/float(w_error))
			weights = np.multiply(weights, np.exp([float(x)*alpha for x in yh]))

			predict_train_iter = [sum(x) for x in zip(predict_train_iter, [float(x)*alpha for x in h_train])]
			predict_test_iter = [sum(x) for x in zip(predict_test_iter, [float(x)*alpha for x in h_test])]

		prediction_train, prediction_test = np.sign(predict_train_iter), np.sign(predict_test_iter)
		return prediction_train, prediction_test

	def create_stump(self, train, weights):
		stump = decision_tree(train, self.attributes, self.attribute_values, depth=1, weights=weights)
		return stump

	def get_predictions(self):
		return self.train_pred, self.test_pred

class BaggedTrees():
	def __init__(self, train, test, attributes, attribute_values, num_trees, sample_size):
		self.num_trees,  self.sample_size = num_trees, float(sample_size)
		self.attributes = attributes
		self.train_pred = []
		self.test_pred = []
		self.attribute_values = attribute_values
		
		# self.bagged(train, test)
		self.train_pred, self.test_pred = self.bagged(train, test)


	def get_predictions(self):
		return self.train_pred, self.test_pred

	def subset(self, data):
		# subset = np.empty(shape=data.shape, dtype=None)
		# print list(data)
		subset = pd.DataFrame(columns=list(data))
		# print 'columns'
		# print list(subset)
		# print 'sample_size = ' + str(self.sample_size)
		num_sub = self.sample_size
		# print 'subset size = ' + str(subset.shape[0])
		i = 0
		while subset.shape[0] < num_sub:
			index = randrange(data.shape[0])
			# print 'index = ' + str(index)
			# print 'num_sub = ' + str(num_sub)
			# print 'current index = ' + str(i)
			# print data.iloc[[index]]
			subset.loc[i] = data.iloc[index]
			i += 1
		# df = pd.DataFrame(subset, columns=self.attributes)
		# print 'subset size = ' + str(subset.shape[0])
		return subset

	def bagged(self, train, test):
		trees = list()
		for i in range(int(self.num_trees)):
			# print 'subset begun'
			# subset = self.subset(train)
			# print 'subset ended'
			subset = pd.DataFrame(columns=list(train))
			print 'current tree = ' + str(i)
			subset = train.sample(n=int(self.sample_size), replace=True)
			subset = subset.reset_index(drop=True)
			tree = decision_tree(subset, self.attributes, self.attribute_values)
			trees.append(tree)
		# 	# pprint.pprint(tree.get_tree())
		prediction_train = self.bag_predict(trees, train)
		# # print 'test'
		prediction_test = self.bag_predict(trees, test)
		return prediction_train, prediction_test

	def bag_predict(self, trees, data):
		predictions = list()
		index = 0
		while index < len(data.index):
			instance = data.iloc[ index , : ]
			predict_result = self.predict(trees, instance)
			predictions.append(predict_result)
			index += 1
		return predictions

	def predict(self, trees, instance):
		predictions = list()
		for tree in trees:
			predict_result = tree.predict(instance)
			predictions.append(predict_result)
		return max(set(predictions), key=predictions.count)


class RandomForest():
	def __init__(self, train, test, attributes, attribute_values, num_trees, sample_size, feature_sample_size, unknown=False):
		self.num_trees = num_trees
		self.sample_size, self.feature_sample_size = float(sample_size), int(feature_sample_size)
		self.attributes = attributes
		self.attribute_values = attribute_values

		self.train_pred, self.test_pred = self.rand_forest(train, test)

	def subset(self, data):
		# subset = np.empty(shape=data.shape, dtype=None)
		# print list(data)
		subset = pd.DataFrame(columns=list(data))
		# print 'columns'
		# print list(subset)
		# print 'sample_size = ' + str(self.sample_size)
		num_sub = self.sample_size
		# print 'subset size = ' + str(subset.shape[0])
		i = 0
		while subset.shape[0] < num_sub:
			index = randrange(data.shape[0])
			# print 'index = ' + str(index)
			# print 'num_sub = ' + str(num_sub)
			# print 'current index = ' + str(i)
			# print data.iloc[[index]]
			subset.loc[i] = data.iloc[index]
			i += 1
		# df = pd.DataFrame(subset, columns=self.attributes)
		# print 'subset size = ' + str(subset.shape[0])
		return subset
	
	def rand_forest(self, train, test):
		trees = list()

		for attribute in self.attributes:
			self.attribute_values[attribute] = np.unique(train[attribute])
		for i in range(int(self.num_trees)):

			subset = pd.DataFrame(columns=list(train))
			print 'current tree = ' + str(i)
			subset = train.sample(n=int(self.sample_size), replace=True)
			subset = subset.reset_index(drop=True)
			
			tree = self.RandTreeLearn(subset, self.attributes)
			trees.append(tree)
			# pprint.pprint(tree)
		prediction_train = self.forest_predict(trees, train)
		prediction_test = self.forest_predict(trees, test)
		return prediction_train, prediction_test

	def RandTreeLearn(self, data, attributes, tree=None):

		features = attributes
		if len(features) >= self.feature_sample_size:
			while features == attributes:
				features = random.sample(attributes, self.feature_sample_size)
		
		best = self.best_split(data, features)
		if tree is None:
			tree = dict()
			tree[best] = dict()

		# print '\n\nbest = ' + best
		for value in self.attribute_values[best]:

			subtree = data[data[best] == value].reset_index(drop=True)
			labels,unique_values = np.unique(subtree['label'], return_counts=True)

			if len(unique_values) == 1: # all examples have the same label
				tree[best][value] = labels[0]
			elif value not in np.unique(data[best]) or (len(features) == 1 and features[0] == best): # Sv is empty
				# get most common label in S
				curr_sub_tree_labels,curr_sub_tree_unique_values = np.unique(data['label'], return_counts=True)
				index = np.unravel_index(np.argmax(curr_sub_tree_unique_values, axis=None), curr_sub_tree_unique_values.shape)
				tree[best][value] = curr_sub_tree_labels[index]
			elif len(features) == 0: # attributes empty
				# get most common label in dataset
				attribute_val_tree = data[data[best] == value]
				new_labels,new_unique_values = np.unique(attribute_val_tree['label'], return_counts=True)
				index = np.unravel_index(np.argmax(new_unique_values, axis=None), new_unique_values.shape)
				tree[best][value] = new_labels[index]
			else:
				temp_list = list(attributes)
				temp_list.remove(best)
				tree[best][value] = self.RandTreeLearn(subtree,temp_list)
		
		return tree

	def forest_predict(self, trees, data):
		predictions = list()
		index = 0
		while index < len(data.index):
			instance = data.iloc[ index , : ]
			predict_result = self.predict(trees, instance)
			predictions.append(predict_result)
			index += 1
		return predictions

	def predict(self, trees, instance):
		predictions = list()
		for tree in trees:
			predict_result = self.tree_predict(instance, tree)
			predictions.append(predict_result)
		return max(set(predictions), key=predictions.count)

	def tree_predict(self, instance, tree):

		for nodes in tree.keys():
			value = instance[nodes]
			sub_tree = tree[nodes][value]
			prediction = 0
	            
			if type(sub_tree) is dict:
				prediction = self.tree_predict(instance, sub_tree)
			else:
				prediction = sub_tree
				break;                            
	        
		return prediction

	def entropy_label(self, data):

		entropy_node = 0  
		values = data.label.unique() 

		for value in values:
			fraction = float(data.label.value_counts()[value])/len(data.label) 
			entropy_node += -fraction*np.log2(fraction)
		return entropy_node

	def entropy(self, data, attribute):

		variables = data[attribute].unique()
		entropy_attribute = 0
		values = data.label.unique()
		for variable in variables:
			entropy_value = 0
			for value in values:
				num = len(data[attribute][data[attribute]==variable][data.label == value])
				den = len(data[attribute][data[attribute]==variable])
				fraction = float(num)/(den+eps) 
				if fraction == 1.0 or fraction == 0.0:
					entropy_value += 0.0
				else:
					entropy_value += -fraction*np.log2(fraction) 
				
			fraction2 = float(den)/len(data)
			entropy_attribute += fraction2*entropy_value

		return entropy_attribute

	def best_split(self, data, attributes):
		info_gain = []

		label_value = self.entropy_label(data)
		
		for attribute in attributes:
			weighted_att_value = self.entropy(data, attribute)

			info_gain.append(label_value - weighted_att_value)

		return attributes[np.argmax(info_gain)]

	def get_predictions(self):
		return self.train_pred, self.test_pred

