import random
import numpy as np
import networkx
import genData
import matplotlib.pyplot as plt

class Perceptron(object):
	""" 
	"""
	def __init__(self, alpha, dimension = 2):
		self.use_kernel_ = False
		self.weight_ = np.array([0,0])
		self.learning_rate_ = alpha
		self.max_iter = 100

	def loadDataFromFile(self, filename):
		self.data_ = np.load(filename)

	def generateRandomData(self, shape, n):
		self.data_ = genData.random_dataset(shape, n)

	def update_weight(self, x_vec):
		y = self.response(x_vec)
		self.weight_ = np.add (self.weight_, self.learning_rate_ * (x_vec[2] - y) * x_vec[0:2])

	def response(self, x_vec):
		y = np.multiply(self.weight_, x_vec[0:2])
		# print y
		if np.sum(y) >= 0:
			return 1
		else:
			return -1
	def fit(self):
		learned = False
		iteration = 0
		while not learned:
			globalError = 0

			for x in self.data_:
				r = self.response(x)
				if x[2] != r:
					self.update_weight(x)
					globalError += abs(x[2] - r)
			iteration += 1
			if globalError == 0 or iteration > self.max_iter:
				learned = True

	def plot_separation(self):
		for x in self.data_:
			if x[2] == 1:
				plt.plot(x[0], x[1], 'ob')
			else:
				plt.plot(x[0], x[1], 'or')
		unit_vec = self.weight_/np.linalg.norm(self.weight_)
		p1 = [unit_vec[1], -unit_vec[0]]
		p2 = [- unit_vec[1], unit_vec[0]]
		plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-.')
		plt.show()





