import math
import numpy as np
from numpy import linalg

def ker(x, y, opt, r=2):
	if opt == 0:
		return gaussian(x,y, r)
	if opt == 1:
		return polynomial(x,y, r)
	if opt == 2:
		return distance(x,y)
	if opt == 3:
		return demo(x,y)
	else:
		return linear_kernel(x,y)

def linear_kernel(x,y):
	return np.dot(x,y)

def gaussian(x, y, sigma):
	""" x,y are numpy vectors
	"""
	gaussian = np.exp(-linalg.norm(x-y)**2/(2 * (sigma ** 2)))
	return gaussian

def polynomial(x, y,p):
	""" x, y are numpy vectors
	"""
	dotp = np.dot(x,y)
	poly = math.pow((dotp + 1), p)
	return poly

def distance(x,y):
	d = linalg.norm(x-y)
	return d

def demo(x,y):
	d = np.dot(x,y)
	nx = linalg.norm(x)
	ny = linalg.norm(y)
	return d + (nx**2)*(ny**2)
