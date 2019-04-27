# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import os
import matplotlib.pyplot as plt
import timeit

class tensor():
	def __init__(self, dim, num_cat, width, depth, num_epoch):

		self.dim, self.num_cat, self.width, self.depth, self.num_epoch = dim, num_cat, width, depth, num_epoch

	def create_model(self, depth, act):
		activation = ''
		initializer = ''
		model = keras.Sequential()

		print('dimension = {}'.format(self.dim))

		if act == 'tanh':
			activation = tf.nn.tanh
			initializer = tf.keras.initializers.glorot_normal
		elif act == 'relu':
			activation = tf.nn.relu
			initializer = tf.keras.initializers.glorot_normal
		else:
			print('invalid activation type')
			exit()

		if depth == 3:
			model = keras.Sequential([
		    keras.layers.Dense(self.width, input_shape=(4,), activation=activation, kernel_initializer=initializer),
		    keras.layers.Dense(self.width, activation=activation, kernel_initializer=initializer),
		    keras.layers.Dense(self.num_cat, activation=tf.nn.softmax)
			])
		elif depth == 5:
			model = keras.Sequential([
		    
		    keras.layers.Dense(self.width, input_shape=(4,), activation=activation, kernel_initializer=initializer),
		    keras.layers.Dense(self.width, activation=activation, kernel_initializer=initializer),
		    keras.layers.Dense(self.width, activation=activation, kernel_initializer=initializer),
		    keras.layers.Dense(self.width, activation=activation, kernel_initializer=initializer),
		    keras.layers.Dense(self.num_cat, activation=tf.nn.softmax)
			])
		elif depth == 9:
			model = keras.Sequential([
		    
		    keras.layers.Dense(self.width, input_shape=(4,), activation=activation, kernel_initializer=initializer),
		    keras.layers.Dense(self.width, activation=activation, kernel_initializer=initializer),
		    keras.layers.Dense(self.width, activation=activation, kernel_initializer=initializer),
		    keras.layers.Dense(self.width, activation=activation, kernel_initializer=initializer),
		    keras.layers.Dense(self.width, activation=activation, kernel_initializer=initializer),
		    keras.layers.Dense(self.width, activation=activation, kernel_initializer=initializer),
		    keras.layers.Dense(self.width, activation=activation, kernel_initializer=initializer),
		    keras.layers.Dense(self.width, activation=activation, kernel_initializer=initializer),
		    keras.layers.Dense(self.num_cat, activation=tf.nn.softmax)
			])
		else:
			print('invalid depth [3, 5, 9]')
			exit()

		model.compile(optimizer=keras.optimizers.Adam(), 
	              loss=tf.keras.losses.sparse_categorical_crossentropy,
	              metrics=['accuracy'])

		return model

	def train_and_evaluate(self, activation, train_data, train_labels, test_data, test_labels):

		model = self.create_model(self.depth, activation)
		model.fit(train_data, train_labels, epochs=self.num_epoch)

		train_loss, train_acc = model.evaluate(train_data, train_labels)
		test_loss, test_acc = model.evaluate(test_data, test_labels)

		print("Trained model, training dataset accuracy: {:5.2f}%".format(100*train_acc))
		print("Trained model, testing dataset accuracy: {:5.2f}%".format(100*test_acc))

