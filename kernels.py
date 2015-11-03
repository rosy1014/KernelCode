import math
import numpy as np


def ker(x, y, opt, r=2):
	if opt == 0:
		return gaussian(x,y, r)
	if opt == 1:
		return polynomial(x,y)
	else:
		return np.dot(x,y)

def gaussian(x, y, r):
	""" x,y are numpy vectors
	"""
	diff = np.subtract(x, y)
	distance = math.pow(np.linalg.norm(diff), 2)
	gaussian = math.exp(- distance / (2.0 * math.pow(r, 2)))
	return gaussian

def polynomial(x, y):
	""" x, y are numpy vectors
	"""
	dotp = np.dot(x,y)
	poly = math.pow((dotp + 1), 2)
	return poly
