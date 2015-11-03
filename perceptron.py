import random
import numpy as np
import networkx
import genData
import matplotlib.pyplot as plt
import kernels

class Perceptron(object):
	""" 
	"""
	def __init__(self, alpha, dimension = 2):
		self.use_kernel_ = False
		self.weight_ = np.array([0,0])
		self.learning_rate_ = alpha
		self.max_iter = 100
		self.kernel_option_ = 0 

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

		n = self.data_.shape[1]
		if not self.use_kernel_:
			print "not using a kernel"
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
		else:
			print "Using a kernel"
			mistake_counter = np.zeros(self.data_.shape[1])
			learned = False
			iteration = 0
			while not learned:
				globalError = 0
				for i in xrange(n):
					xi = self.data_[i]
					r = 0
					for j in xrange(n):
						xj = self.data_[i]
						r += mistake_counter[i] * xi[2] * self.kernel(xi[0:2], xj[0:2])
					if r != xi[2]:
						mistake_counter[i] += 1
						globalError += 1
				iteration += 1
				if globalError == 0 or iteration > self.max_iter:
					for i in xrange(n):
						self.weight_ = np.add(mistake_counter[i] * self.data_[i,2] * self.data_[i, 0:2], self.weight_)
					learned = True


	def choose_kernel_option(self, kernel_option):
		self.kernel_option_ = kernel_option
		self.use_kernel_ = True

	def kernel(self, x1, x2):
		return kernels.ker(x1, x2, self.kernel_option_)

	def use_kernel(self, kernel_option):
		""" kernel_option=0: gaussian kernel
			kernel_option=1: polynomial kernel
		"""
		n = self.data_.shape[1]
		K = np.empty([n,n])
		for i in xrange(n):
			for j in xrange(i, n):
				K[i,j] = kernels.ker(self.data_[0:2,i], self.data_[0:2,j], kernel_option)
		K = K + K.transpose() - np.diag(K)
		print K
		self.use_kernel_ = True
		return K

		self.kernel_matrix_ = self.data_
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

	def train_test_split(self, ratio):
		shuffled_data = np.random.shuffle(self.data_)
		n = self.data_.shape[1]
		train_size = n * ratio
		self.data_ = shuffled_data[:n*ratio]

	def test_classifier(self):
		predict11 = [] # correct pink
		predict12 = [] # wrong red
		predict21 = [] # wrong blue
		predict22 = [] # correct green
		for x in self.data_:
			r = np.dot(self.weight_, x[0:2])
			if r > 0 and x[2] > 0:
				predict11.append(x)
			if r < 0 and x[2] > 0:
				predict12.append(x)
			if r > 0 and x[2] < 0:
				predict21.append(x)
			if r < 0 and x[2] < 0:
				predict22.append(x)
		print "number of false negatives:"
		print len(predict12)
		print "number of false positives:"
		print len(predict21)
		
		for x in predict11:
			plt.plot(x[0], x[1], 'om')
		for x in predict12:
			plt.plot(x[0], x[1], 'or')
		for x in predict21:
			plt.plot(x[0], x[1], 'ob')
		for x in predict22:
			plt.plot(x[0], x[1], 'oc')
		plt.show()






