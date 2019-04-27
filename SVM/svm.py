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
from scipy.optimize import minimize
from random import randrange

class SVM():
	def __init__(self, learning_rate, learning_rate_tweak, hyper_param, num_epoch):

		self.r, self.d, self.c, self.epoch = learning_rate, learning_rate_tweak, hyper_param, num_epoch

	def primal(self, train):
		
		weights = list()
		w_i = list()
		w_i = np.zeros(train.shape[1] - 1)
		weights.append(w_i)
		self.gradients = list()
		j = list()

		weight_idx = 0
		for i in range(self.epoch):
			
			np.random.shuffle(train)
			idx = 0
			for x in train:
				lr = 0.0
				if self.d != 0:
					lr = self.r / (1 + (self.r / self.d) * idx)
				else:
					lr = self.r / (1 + idx)
				w_i = weights[weight_idx]
				grad = 0.0
				if (x[-1] * np.dot(w_i, x[:-1])) <= 1.0:
					grad = w_i - self.c*x[-1]*x[:-1]
				else:
					grad = w_i
				
				j.append(0.5*np.dot(w_i, w_i) + self.c*max(0, 1 - x[-1]*np.dot(w_i, x[:-1])))
				weights.append(w_i - lr*grad)
				weight_idx = weight_idx + 1
				self.gradients.append(grad)
				idx = idx + 1

		return weights[-1], j

	def test_accuracy(self, data, weights):
		pred_acc = 0.0
		num_correct = 0.0

		for x in data:
			pred = self.prediction(weights, x[:-1])
			if pred == x[-1]:
				num_correct = num_correct + 1

		pred_acc = num_correct / len(data)
		return pred_acc

	def test_dual_acc(self, data, alpha, sup_vec, kernel):
		pred_acc = 0.0
		num_correct = 0.0

		for x in data:
			pred = self.dual_prediction(alpha, sup_vec, x[:-1], kernel)
			if pred == x[-1]:
				num_correct = num_correct + 1

		pred_acc = num_correct / len(data)
		return pred_acc

	def dual_prediction(self, alpha, sup_vec, instance, kernel):
		x = list()
		y = list()
		for x_i in sup_vec:
			x.append(x_i[:-1])
			y.append(x_i[-1])
		if kernel:
			return np.sign(sum(alpha[i]*y[i]*self.kernel(x[i], instance, self.r) for i in range(len(x))))
		else:
			return np.sign(sum(alpha[i]*y[i]*np.dot(x[i], instance) for i in range(len(x))))

	def prediction(self, weights, instance):
		return np.sign(np.dot(weights, instance))

	def get_gradients(self):
		return self.gradients

	def dual_objective(self, alpha):
		return 0.5*np.dot(np.dot(alpha, self.H), alpha) - np.dot(np.ones(len(alpha)), alpha)

	def constraint(self, alpha):
		return np.dot(self.y, alpha)

	def kernel(self, x, z, c):
		return np.exp(-1*((np.linalg.norm(x) - np.linalg.norm(z)) ** 2 / c))

	def dual_minimization(self, train, kernel):

		alpha = np.zeros(len(train))
		H = list()
		x = list()
		y = list()

		for x_i in train:
			x.append(x_i[:-1])
			y.append(x_i[-1])

		for i in range(len(x)):
			row = list()
			for j in range(len(x)):
				if kernel:
					kernel_i_j = self.kernel(x[i], x[j], self.r)
					row.append(y[i]*y[j]*kernel_i_j)
				else:
					row.append(y[i]*y[j]*np.dot(x[i], x[j]))
			H.append(row)
			
		H = np.array(H)

		self.H = H
		self.x = x
		self.y = y

		b = (0, self.c)
		bnds = list()
		for i in range(len(alpha)):
			bnds.append(b)
		con = {'type': 'eq', 'fun':self.constraint}
		print "running minimization"
		sol = minimize(self.dual_objective, alpha, method='SLSQP', bounds=bnds, constraints=con)
		print 'finished'

		return sol.x






