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

class Perceptron():
	def __init__(self, train, test, learning_rate, num_epoch, percept_type):

		self.r, self.epoch = learning_rate, num_epoch
		self.weight_vec = list()
		self.avg = list()
		self.cost = list()

		if percept_type == 'standard':
			self.weight_vec = self.percept_stand(train)
		elif percept_type == 'voted':
			self.weight_vec, self.cost = self.percept_vote(train)
		else:
			self.avg, self.weight_vec = self.percept_avg(train)

		self.pred_acc = self.test_accuracy(test, percept_type)

	def get_results(self):
		return self.weight_vec, self.pred_acc

	def get_voted_results(self):
		return self.weight_vec, self.cost, self.pred_acc

	def test_accuracy(self, data, percept_type):
		num_correct = 0

		for example in data:
				y = example[-1]
				x = example[:-1]
				pred = self.prediction(x, percept_type)
				if pred == y:
					num_correct = num_correct + 1

		print len(data)
		return float(num_correct) / len(data)


	def prediction(self, instance, percept_type):

		pred = 0.0
		if percept_type == 'standard':
			pred = np.sign(np.dot(self.weight_vec, instance))
		elif percept_type == 'voted':
			for i in range(len(self.weight_vec)):
				pred = pred + self.cost[i] * np.sign(np.dot(self.weight_vec[i], instance))
			pred = np.sign(pred)
		else:
			pred = np.sign(np.dot(self.avg, instance))

		return pred

	def percept_stand(self, train):
		weights = list()
		weights = np.zeros(train.shape[1] - 1)

		for i in range(self.epoch):
			np.random.shuffle(train)
			for x in train:
				if (x[-1] * np.dot(weights, x[:-1])) <= 0.0:
					weights = weights + self.r * x[-1]*x[:-1]

		return weights

	def percept_vote(self, train):
		weights = list()
		cost = list()

		m = 0
		for i in range(m + 2):
			w = np.zeros(train.shape[1] - 1)
			weights.append(w)
			cost.append(0.0)

		for i in range(self.epoch):

			for x in train:
				# print m
				# print len(weights)
				if (x[-1] * np.dot(weights[m], x[:-1])) <= 0.0:
					weights[m + 1] = weights[m] + self.r * x[-1]*x[:-1]
					m = m + 1
					cost[m] = 1
					if m >= (len(weights) - 1):
						weights.append(np.zeros(train.shape[1] - 1))
						cost.append(0.0)
				else:
					cost[m] = cost[m] + 1

		return weights, cost

	def percept_avg(self, train):
		weights = list()
		weights = np.zeros(train.shape[1] - 1)
		avg = list()
		avg = np.zeros(train.shape[1] - 1)

		for i in range(self.epoch):

			for x in train:
				if (x[-1] * np.dot(weights, x[:-1])) <= 0.0:
					weights = weights + self.r * x[-1]*x[:-1]
				avg = avg + weights

		return avg, weights






