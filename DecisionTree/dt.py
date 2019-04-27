import numpy as np 
import pandas as pd 
eps = np.finfo(float).eps
inf = np.finfo(float).max
import sys, getopt
import argparse
import csv
import pprint

class decision_tree():
	def __init__(self, train, attributes, attribute_values, gain_type='entropy', depth=str(inf), unknown=False, weights=None):
		self.gain_type, self.depth, self.weights = gain_type, depth, weights
		self.attributes = attributes
		self.attribute_values = attribute_values
		
		if weights is not None:
			train['weight'] = weights

		self.tree = self.ID3(train, self.attributes, 0)

	def entropy_label(self, data):

		entropy_node = 0  
		values = data.label.unique() 

		for value in values:
			fraction = 0.0
			if self.weights is not None:
				num = data['weight'][data.label == value].sum(axis=0)
				fraction = float(num)/len(data.label)
			else:
				fraction = float(data.label.value_counts()[value])/len(data.label) 
			entropy_node += -fraction*np.log2(fraction)
		return entropy_node

	def gini_label(self, data):

		gini_node = 0
		values = data.label.unique()
		for value in values:
			
			fraction = float(data.label.value_counts()[value])/len(data.label) 
			gini_node += fraction ** 2
		gini_node = 1 - gini_node
		return gini_node

	def me_label(self, data):

		me_node = 0
		values = data.label.unique()
		for value in values:
			fraction = float(data.label.value_counts()[value])/len(data.label) 
			if fraction > me_node:
				me_node = fraction
		me_node = 1 - me_node
		return me_node

	def entropy(self, data, attribute):

		variables = data[attribute].unique()
		entropy_attribute = 0
		values = data.label.unique()
		for variable in variables:
			entropy_value = 0
			for value in values:
				num = 0.0
				if self.weights is not None:
					num = data['weight'][data[attribute] == variable][data.label == value].sum(axis=0)
				else:
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

	def gini(self, data, attribute):

		variables = data[attribute].unique()
		gini_attribute = 0
		values = data.label.unique()
		for variable in variables:
			gini_value = 0
			for value in values:
				num = len(data[attribute][data[attribute]==variable][data.label == value])
				den = len(data[attribute][data[attribute]==variable])
				fraction = float(num)/(den+eps)
				gini_value += fraction ** 2
			gini_value = 1 - gini_value
			fraction2 = float(den)/len(data)
			gini_attribute += fraction2*gini_value

		return gini_attribute

	def me(self, data, attribute):
		
		variables = data[attribute].unique()
		me_attribute = 0
		values = data.label.unique()
		for variable in variables:
			me_value = 0
			for value in values:
				num = len(data[attribute][data[attribute]==variable][data.label == value])
				den = len(data[attribute][data[attribute]==variable])
				fraction = float(num)/(den+eps)
				if fraction > me_value:
					me_value = fraction
			me_value = 1 - me_value
			fraction2 = float(den)/len(data)
			me_attribute += fraction2*me_value
		
		return me_attribute

	def best_split(self, data, attributes):
		info_gain = []

		label_value = 0
		if self.gain_type == 'entropy':
			label_value = self.entropy_label(data)
		elif self.gain_type == 'gini':
			label_value = self.gini_label(data)
		elif self.gain_type == 'majority_error':
			label_value = self.me_label(data)
		
		for attribute in attributes:

			weighted_att_value = 0
			if self.gain_type == 'entropy':
				weighted_att_value = self.entropy(data, attribute)
			elif self.gain_type == 'gini':
				weighted_att_value = self.gini(data, attribute)
			elif self.gain_type == 'majority_error':
				weighted_att_value = self.me(data, attribute)

			info_gain.append(label_value - weighted_att_value)

		return attributes[np.argmax(info_gain)]

	def ID3(self, data, attributes, curr_depth, tree=None):
		best = self.best_split(data, attributes)

		if tree is None:
			tree = dict()
			tree[best] = dict()

		# print '\n\nbest = ' + best
		for value in self.attribute_values[best]:

			subtree = data[data[best] == value].reset_index(drop=True)
			labels,unique_values = np.unique(subtree['label'], return_counts=True)
			if len(unique_values) == 1: # all examples have the same label
				tree[best][value] = labels[0]
			elif value not in np.unique(subtree[best]) or (len(attributes) == 1 and attributes[0] == best): # Sv is empty
				# get most common label in S
				curr_sub_tree_labels,curr_sub_tree_unique_values = np.unique(data['label'], return_counts=True)
				index = np.unravel_index(np.argmax(curr_sub_tree_unique_values, axis=None), curr_sub_tree_unique_values.shape)
				tree[best][value] = curr_sub_tree_labels[index]
			elif len(attributes) == 0: # attributes empty
				# get most common label in dataset
				attribute_val_tree = data[data[best] == value]
				new_labels,new_unique_values = np.unique(attribute_val_tree['label'], return_counts=True)
				index = np.unravel_index(np.argmax(new_unique_values, axis=None), new_unique_values.shape)
				tree[best][value] = new_labels[index]
			elif float(curr_depth) >= float(self.depth): # max depth reached
				new_labels,new_unique_values = np.unique(data['label'], return_counts=True)
				index = np.unravel_index(np.argmax(new_unique_values, axis=None), new_unique_values.shape)
				tree[best][value] = new_labels[index]
			else:
				temp_list = list(attributes)
				temp_list.remove(best)
				tree[best][value] = self.ID3(subtree,temp_list,curr_depth + 1)
		
		return tree

	def predict(self, instance, tree=None):

		if tree is None:
			tree = self.tree

		for nodes in tree.keys():
			value = instance[nodes]
			sub_tree = tree[nodes][value]
			prediction = 0
	            
			if type(sub_tree) is dict:
				prediction = self.predict(instance, sub_tree)
			else:
				prediction = sub_tree
				break;                            
	        
		return prediction

	def get_tree(self):
		return self.tree
	
	

