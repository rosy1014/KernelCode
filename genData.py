import random
import numpy as np
import math
import matplotlib.pyplot as plt

def random_dataset (shape,  n, dimension=2,):
	""" Shape can "cc", "rt", "cs", "ga", "rn"
		right now, the first 4 options have not been implemented.
	"""
	if shape == "cc":
		return gen_two_circles(n)
	elif shape == "rt":
		return gen_two_rectangles(n)
	elif shape == "cs":
		return gen_two_curly_separable(n)
	elif shape == "ga":
		return gen_embeded(n)
	else:
		return gen_two_randoms(n)


def gen_two_circles(n, dimension=2):
	pos = np.random.rand(n) 
	pos_cos = np.cos(pos * math.pi * 2)
	pos_sin = np.sin(pos * math.pi * 2)
	# print pos

	neg = np.random.rand(n)
	# print neg
	neg_cos = np.cos(neg * math.pi * 2)
	neg_sin = np.sin(neg * math.pi * 2)
	# print neg
	rp = np.random.rand(n)
	rn = np.random.rand(n)
	xp = np.multiply(pos_cos, rp) + 1
	yp = np.multiply(pos_sin, rp) + 0.5
	xn = np.multiply(neg_cos, rn) - 1
	yn = np.multiply(neg_sin, rn) + 0.5
	# xp = pos + 0.5
	# yp = pos + 0.5
	# xn = neg -0.5
	# yn = neg + 0.5
	inputs = []
	for i in range(n):
		inputs.append([xp[i], yp[i], 1])
		inputs.append([xn[i], yn[i], -1])

	return np.array(inputs)




def gen_two_rectangles(n, dimension=2):
	xp = (np.random.rand(n) * 2 -1)/2
	yp = (np.random.rand(n) * 2 -1)/2
	xn = (np.random.rand(n) * 2 -1)/2
	yn = (np.random.rand(n) * 2 -1)/2
	# xp = pos + 0.5
	# yp = pos + 0.5
	# xn = neg -0.5
	# yn = neg + 0.5
	inputs = []
	for i in range(n):
		inputs.append([xp[i] + 0.6, yp[i] + 0.5, 1])
		inputs.append([xn[i] - 0.6, yn[i] + 0.5, -1])

	return np.array(inputs)



def gen_two_curly_separable(n, dimension=2):
	return 0


def gen_embeded(n, dimension=2):
	pos = np.random.rand(n) 
	pos_cos = np.cos(pos * math.pi * 2)
	pos_sin = np.sin(pos * math.pi * 2)
	# print pos

	neg = np.random.rand(n)
	# print neg
	neg_cos = np.cos(neg * math.pi * 2)
	neg_sin = np.sin(neg * math.pi * 2)

	rp = np.random.rand(n) + 1.5 
	rn = np.random.rand(n)
	xp = np.multiply(pos_cos, rp) + 1
	yp = np.multiply(pos_sin, rp) + 0.5
	xn = np.multiply(neg_cos, rn) + 1
	yn = np.multiply(neg_sin, rn) + 0.5	

	inputs = []
	for i in range(n):
		inputs.append([xp[i], yp[i], 1])
		inputs.append([xn[i], yn[i], -1])
    
	return np.array(inputs)	
	

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

