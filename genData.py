import random
import numpy as np

import matplotlib.pyplot as plt

def random_dataset (shape,  n, dimension=2,):
	""" Shape can "cc", "rt", "cs", "mn", "rn"
		right now, the first 4 options have not been implemented.
	"""
	if shape == "cc":
		return gen_two_circles(dimension, n)
	elif shape == "rt":
		return gen_two_rectangles(dimension, n)
	elif shape == "cs":
		return gen_two_curly_separable(dimension, n)
	elif shape == "mn":
		return gen_two_moons(dimension, n)
	else:
		return gen_two_randoms(dimension, n)


def gen_two_circles(n, dimension=2):
	return 0



def gen_two_rectangles(n, dimension=2):
	return 0


def gen_two_curly_separable(n, dimension=2):
	return 0


def gen_two_moons(n, dimension=2):
	return 0

def gen_two_randoms(n, dimension=2):
	""" generates a linearly separable dataset with n samples 
	return 0
	"""
	xp = (np.random.rand(n)*2 - 1)/2 - 0.5
	yp = (np.random.rand(n)*2 - 1)/2 + 0.5
	xn = (np.random.rand(n)*2 - 1)/2 + 0.5
	yn = (np.random.rand(n)*2 - 1)/2 - 0.5
	inputs = []
	for i in range(n):
		inputs.append([xp[i], yp[i], 1])
		inputs.append([xn[i], yn[i], -1])

	return np.array(inputs)



def save_to_file(data, filename):
	np.save(filename, data)

