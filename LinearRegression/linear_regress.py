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
from random import randrange
from numpy import linalg as LA
from DecisionTree.dt import decision_tree

class gradient():
	def __init__(self, train_data, learning_rate, num_iter, grad_type='batch',thresh=1e-6):
		self.num_iter = num_iter
		self.weight_vec = list()
		self.bias = 0
		self.cost_fct = list()

		if grad_type == 'batch':
			self.weight_vec, self.bias, self.cost_fct = self.batch(train_data, thresh, learning_rate, num_iter)
		else:
			self.weight_vec, self.bias, self.cost_fct = self.stoch(train_data, thresh, learning_rate, num_iter)

	def get_results(self):
		return self.weight_vec, self.bias, self.cost_fct

	def batch(self, train_data, thresh, learning_rate, num_iter):
		weights = list()
		bias = 0
		cost_fct = list()
		weight_diff_norm = 1.0
		weights = np.zeros(train_data.shape[1] - 1)
		cost_diff = 1.0
		init_cost = self.calc_cost(train_data, weights, bias)
		cost_fct.append(init_cost)
		while weight_diff_norm > 1e-6:
			grad = np.zeros(train_data.shape[1] - 1)

			for example in train_data:
				y = example[-1]
				row = example[:-1]
				for index in range(len(row)):
					grad[index] += (y - (np.dot(weights,row) + bias))*float(row[index])
			
			grad = grad * -1	
			new_weight = weights - float(learning_rate)*grad
			weight_diff = new_weight - weights
			weight_diff_norm = np.dot(weight_diff, weight_diff)
			new_bias = bias - float(learning_rate)*(y - np.dot(weights,row) + bias)
			weights = new_weight
			bias = new_bias
			j = self.calc_cost(train_data, new_weight, new_bias)
			cost_diff = abs(cost_fct[-1] - j)
			cost_fct.append(j)
			

		return weights, bias, cost_fct

	def calc_cost(self, data, weight, bias):
		cost = 0.0
		for index in range(data.shape[0]):
			y = data[index][-1]
			row = data[index][:-1]
			current_cost = (y - (np.dot(weight, row) + bias))**2
			cost += float(current_cost)
		cost = cost * 0.5

		return cost


	def stoch(self, train_x, thresh, learning_rate, num_iter):
		weights = list()
		bias = 0
		cost_fct = list()

		weights = np.zeros(train_x.shape[1] - 1)
		cost_diff = 1.0
		init_cost = self.calc_cost(train_x, weights, bias)
		cost_fct.append(init_cost)
		for i in range(int(num_iter)):
			example = train_x[np.random.randint(train_x.shape[0]), :]
			y = example[-1]
			row = example[:-1]
			new_weight = list()
			for index in range(len(row)):
				w = weights[index] + float(learning_rate)*(y - (np.dot(weights,row) + bias))*float(row[index])
				new_weight.append(w)
			new_bias = bias + float(learning_rate)*(y - np.dot(weights,row) + bias)
			weights = new_weight
			bias = new_bias
			j = self.calc_cost(train_x, new_weight, new_bias)
			cost_diff = abs(cost_fct[-1] - j)
			cost_fct.append(j)
			

		return weights, bias, cost_fct