import numpy as np 
import pandas as pd 
eps = np.finfo(float).eps
inf = np.finfo(float).max
import sys, getopt
import argparse
import pprint
from random import seed
from random import randrange
from csv import reader

class dt():
	def __init__(self, train, test, attributes, attribute_values, gain_type='entropy', depth=str(inf), unknown=False, weights=None):
		self.gain_type, self.depth, self.weights = gain_type, depth, weights
		self.attributes = attributes
		self.attribute_values = attribute_values
		
		if weights is not None:
			train['weight'] = weights

		train_data = self.load_csv(train)
		test_data = self.load_csv(test)
		# for i in range(len(train_data[0])):
		# 	self.str_column_to_float(train_data, i)

		# for i in range(len(test_data[0])):
		# 	self.str_column_to_float(test_data, i)

		self.tree = self.decision_tree(train_data, test_data, depth)

	def load_csv(self, filename):
		file = open(filename, "rb")
		lines = reader(file)
		dataset = list(lines)
		return dataset

	def str_column_to_float(self, dataset, column):
		for row in dataset:
			row[column] = float(row[column].strip())

	def to_terminal(self, group):
		outcomes = [row[-1] for row in group]
		return max(set(outcomes), key=outcomes.count)

	def test_split(self, index, value, dataset):
		left, right = list(), list()
		for row in dataset:
			if row[index] < value:
				left.append(row)
			else:
				right.append(row)
		return left, right

	def gini_index(self, groups, classes):
		# count all samples at split point
		n_instances = float(sum([len(group) for group in groups]))
		# sum weighted Gini index for each group
		gini = 0.0
		for group in groups:
			size = float(len(group))
			# avoid divide by zero
			if size == 0:
				continue
			score = 0.0
			# score the group based on the score for each class
			for class_val in classes:
				p = [row[-1] for row in group].count(class_val) / size
				score += p * p
			# weight the group score by its relative size
			gini += (1.0 - score) * (size / n_instances)
		return gini

	def get_split(self, dataset):
		class_values = list(set(row[-1] for row in dataset))
		b_index, b_value, b_score, b_groups = 999, 999, 999, None
		for index in range(len(dataset[0])-1):
			for row in dataset:
				groups = self.test_split(index, row[index], dataset)
				gini = self.gini_index(groups, class_values)
				if gini < b_score:
					b_index, b_value, b_score, b_groups = index, row[index], gini, groups
		return {'index':b_index, 'value':b_value, 'groups':b_groups}

	def split(self, node, max_depth, depth):
		left, right = node['groups']
		del(node['groups'])
		# check for a no split
		if not left or not right:
			node['left'] = node['right'] = self.to_terminal(left + right)
			return
		# check for max depth
		if depth >= max_depth:
			node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
			return

	def build_tree(self, train, max_depth):
		root = self.get_split(train)
		self.split(root, max_depth, 1)
		return root

	def predict(self, node, rowclear):
		if row[node['index']] < node['value']:
			if isinstance(node['left'], dict):
				return self.predict(node['left'], row)
			else:
				return node['left']
		else:
			if isinstance(node['right'], dict):
				return self.predict(node['right'], row)
			else:
				return node['right']

	def decision_tree(self, train, test, depth):
		tree = self.build_tree(train, depth)
		predictions = list()
		for row in test:
			prediction = self.predict(tree, row)
			predictions.append(prediction)
		return predictions