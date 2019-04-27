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
from DecisionTree.dt import decision_tree

class RandomForest():
	def __init__(self, train, attribute_list, gain_type, depth, unknown=False):
		tree = decision_tree(train, attribute_list, gain_type, depth, unknown)
		pprint.pprint(tree.get_tree())