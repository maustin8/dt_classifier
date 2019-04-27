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

class NN():
	def __init__(self, learning_rate, learning_rate_tweak, width, num_epoch, input_size, depth = 2, output_size = 1):

		self.r, self.d, self.w, self.epoch, self.input_size = learning_rate, learning_rate_tweak, width, num_epoch, input_size
		self.depth = depth
		self.output_size = output_size

		
		self.network = list()
		self.network.append({'weights':np.random.randn(self.input_size, self.w), 'delta':np.zeros(self.w), 'output':np.zeros((self.input_size, self.w))})
		self.network.append({'weights':np.random.randn(self.w + 1, self.w), 'delta':np.zeros(self.w), 'output':np.zeros((self.w + 1, self.w))})
		self.network.append({'weights':np.random.randn(self.w + 1, self.output_size), 'delta':np.zeros(self.w), 'output':np.zeros((self.w + 1, self.output_size))})
		

		self.cache = list()
		# self.network = list()
		# self.network.append({'weights':np.random.randn(self.input_size, self.w), 'delta':np.zeros(self.w), 'output':np.zeros((self.input_size, self.w))})
		# self.network.append({'weights':np.random.randn(self.w + 1, self.w), 'delta':np.zeros(self.w), 'output':np.zeros((self.w + 1, self.w))})
		# self.network.append('weights':np.random.randn(self.w + 1, self.w), 'delta':np.zeros(self.w), 'output':np.zeros((self.w + 1, self.output_size))})

		# self.grad = list()
		# self.grad.append(np.zeros(self.w - 1))
		# self.grad.append(np.zeros(self.w - 1))
		# self.grad.append(np.zeros(self.w - 1))

		# self.network[2]['delta'][0] = -0.062
		# self.network[2]['delta'][1] = -3.375
		# print('weights')

		# 		for 
		# 	print(w['weights'])
		# print('testing forward propagation')

		# for layer in self.network:
		# 	print('layer weights = ', layer['weights'])
		# 	print('layer delta = ', layer['delta'])
		
		
		self.forward(self.network, [1,1,1])
		
		grad = self.back_prop(1)
		lr = self.r/(1+(self.r/self.d)*1)
		for j in range(len(self.network)):
			layer = self.network[j]
			for k in range(len(layer['weights'])):
				print('orig = {}'.format(layer['weights'][k]))
				print('new = {}'.format(grad[j][k]))
				print('learning_rate applied = {}'.format(lr*grad[j][k]))
				for l in range(len(layer['weights'][k])):
					print('substracting {} by {}'.format(layer['weights'][k][l], lr*grad[j][k][l]))
					layer['weights'][k][l] = float(layer['weights'][k][l]) - lr*grad[j][k][l]
				print('new weights = {}'.format(layer['weights'][k]))

	def get_network(self):
		return self.network

	def back_prop(self, expected):
		output_layer = True
		grad = list()
		grad.append(np.zeros((self.input_size, self.w)))
		grad.append(np.zeros((self.w + 1, self.w)))
		grad.append(np.zeros((self.w + 1, self.output_size)))

		for i in reversed(range(len(self.network))):
			print('i = {}'.format(i))
			layer = self.network[i]
			if i != len(self.network) - 1:
				prev_layer = self.network[i + 1]
				print(prev_layer['delta'])
				layer['delta'] = np.zeros(self.w)
				print(layer['output'])
				print(layer['weights'])
				first = True
				for j in range(len(layer['weights'])):
					for k in range(len(layer['weights'][j])):
						print('weight j:{} k:{} = {}'.format(j, k, layer['weights'][j][k]))
						print('output = {}'.format(layer['output'][k]))
						print('delta = {}'.format(prev_layer['delta'][k]))
						sub_delta = self.sigmoid_derivative(layer['output'][k]) * layer['weights'][j][k]*prev_layer['delta'][k]
						
						print('current gradient j:{} k:{} = '.format(i, j, k, grad[j-1][k]))
						grad[i][j][k] = sub_delta
						print('sub delta = {}'.format(self.sigmoid_derivative(layer['output'][k]) * layer['weights'][j][k]*prev_layer['delta'][k]))
						if first != True:
							layer['delta'][j - 1] += sub_delta
					first = False
					print('new delta = {}'.format(layer['delta']))
					
			else:
				print('expected = {}'.format(expected))
				print('calculated = {}'.format(layer['output']))

				loss = (layer['output'] - expected)
				layer['delta'][0] = loss
				grad[i][0] = loss
				for j in range(len(layer['weights']) - 1):
					layer['delta'][j] = loss * self.network[i - 1]['output'][j]
					grad[i][j + 1] =  layer['delta'][j]
		for i in range(len(grad)):
			print('gradients = {}'.format(grad[i]))
		return grad

	#stochastic gradient descent
	def train(self, train):

		for i in range(self.epoch):
			print('epoch: ', i)
			np.random.shuffle(train)
			lr = self.r/(1+(self.r/self.d)*i)
			for x in train:
				self.forward(self.network, x[:-1])
				grad = self.back_prop(x[-1])
				for j in range(len(self.network)):
					layer = self.network[j]
					for k in range(len(layer['weights'])):
						# print('orig = {}'.format(layer['weights'][k]))
						# print('new = {}'.format(grad[j][k]))
						# print('learning_rate applied = {}'.format(lr*grad[j][k]))
						for l in range(len(layer['weights'][k])):
							# print('substracting {} by {}'.format(layer['weights'][k][l], lr*grad[j][k][l]))
							layer['weights'][k][l] = float(layer['weights'][k][l]) - lr*grad[j][k][l]
						# print('new weights = {}'.format(layer['weights'][k]))

		return self.network

	def testing_accuracy(self, data, weights):
		pred_acc = 0.0
		num_correct = 0.0
		for x in data:
			pred = self.forward(weights, x[:-1])
			if pred == x[-1]:
				num_correct = num_correct + 1

		pred_acc = num_correct / len(data)
		return pred_acc

	def forward(self, network, instance):
		
		z1 = np.array(self.sigmoid(np.dot(instance, network[0]['weights'])))
		self.cache.append(z1)
		network[0]['output'] = z1
		print(z1)

		z2 = np.array(self.sigmoid(network[1]['weights'][0] + np.dot(z1, network[1]['weights'][1:])))
		self.cache.append(z2)
		network[1]['output'] = z2
		print(z2)

		y = np.array(network[2]['weights'][0] + np.dot(z2, network[2]['weights'][1:]))
		self.cache.append(y)
		network[2]['output'] = y
		print(y)

		return y

	def sigmoid_derivative(self, z):
		return self.sigmoid(z)*(1-self.sigmoid(z))

	def sigmoid(self, z):
		return 1/(1+np.exp(-z))







