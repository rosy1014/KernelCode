import random
import numpy as np

import matplotlib.pyplot as plt

def random_dataset (shape,  n, dimension=2,):
	""" Shape can "cc", "rt", "cs", "mn", "rn"
		right now, the first 4 options have not been implemented.
	"""
	if shape == "cc":
		return gen_two_circles(n)
	elif shape == "rt":
		return gen_two_rectangles(n)
	elif shape == "cs":
		return gen_two_curly_separable(n)
	elif shape == "mn":
		return gen_two_moons(n)
	else:
		return gen_two_randoms(n)


def gen_two_circles(n, dimension=2):
	pos = (np.random.rand(n) * 2 - 1)/2 
	print pos
	neg = (np.random.rand(n) * 2 - 1)/2
	print neg
	# xp = pos + 0.5
	# yp = pos + 0.5
	# xn = neg -0.5
	# yn = neg + 0.5
	inputs = []
	for i in range(n):
		inputs.append([pos[i] * random.randint(-1,1) + 0.6, pos[i] * random.randint(-1,1)+0.5, 1])
		inputs.append([neg[i] * random.randint(-1,1) - 0.6, neg[i] * random.randint(-1,1)+0.5, -1])

	return np.array(inputs)




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

