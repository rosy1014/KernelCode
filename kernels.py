import math
import numpy as np

def gaussian(x, y):
	""" x,y are numpy vectors
	"""
	diff = np.subtract(x, y)
	distance = math.pow(np.linalg.norm(diff), 2)
	gaussian = math.exp(- distance / 2.0)
	return gaussian

def polynomial(x, y):
	""" x, y are numpy vectors
	"""
	dotp = np.dot(x,y)
	poly = math.pow((dotp + 1), 2)
	return poly
