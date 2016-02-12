import random
import numpy as np
import networkx
import genData
import matplotlib.pyplot as plt
import kernels
import copy
# import MDS

class Perceptron(object):
	""" 2 dimension Perceptron with kernel
	"""
	def __init__(self, learning_rate, dimension = 2, speed = 1):
		self.learning_rate = learning_rate
		self.max_iter = 100
		self.kernel_option_ = 0 
		self.learning_alpha = []
		self.learning_weight = []
		self.speed = speed

	def loadDataFromFile(self, filename):
		self.data_ = np.load(filename)

	def generateRandomData(self, shape, n):
		self.data_ = genData.random_dataset(shape, n)
		return self.data_

	def update_weight(self, x_vec):
		y = self.response(x_vec)
		self.weight_ = np.add (self.weight_, self.learning_rate * (x_vec[2] - y) * x_vec[0:2])

	def response(self, x_vec):
		y = np.multiply(self.weight_, x_vec[0:2])
		# print y
		if np.sum(y) >= 0:
			return 1
		else:
			return -1

	def fit(self):
		K = self.use_kernel()
		y = np.concatenate(self.data_[:,-1:])
		print y
		n_samples, n_features = self.data_[:, 0:-1].shape
		print "number of features is %d" %n_features
		self.weight_ = np.zeros(n_features, dtype = np.float64)
		self.alpha = np.zeros(n_samples, dtype = np.float64)
		
		learned = False
		iteration = 0
		while not learned:
			globalError = 0
			for i in xrange(n_samples):
				if np.sign(np.sum(np.multiply(np.multiply(K[:,i], self.alpha), y))) != y[i]: # fix dot
					globalError += 1
					self.alpha[i] += self.learning_rate
				temp_alpha = (self.alpha[:], self.data_[i,0:-1])
				self.learning_alpha.append(copy.deepcopy(temp_alpha))
			iteration += 1
			if globalError == 0 or iteration > self.max_iter:
				learned = True

		print "end iteration"
		print iteration

		self.sv = self.data_[:, 0:-1]
		self.sv_y = y


	def predict(self):
		y_predict = np.zeros(len(self.data_))
		for i in xrange(len(self.data_)):
			s = 0
			for a, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
				s += a * sv_y * self.kernel(self.data_[i,:-1], sv)
			y_predict[i] = s
		return np.sign(y_predict)
	
	def predict_weight(self, x1, x2):
	    s = 0
	    for a, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
	    	s += a * sv_y * self.kernel([x1, x2], sv)
	    return s

	def choose_kernel_option(self, kernel_option):
		self.kernel_option_ = kernel_option

	def kernel(self, x1, x2):
		return kernels.ker(x1, x2, self.kernel_option_)

	def use_kernel(self):
		""" kernel_option=0: gaussian kernel
			kernel_option=1: polynomial kernel
			Otherwise:       linear kernel
		"""
		n = self.data_.shape[0]
		K = np.empty([n,n])
		for i in xrange(n):
			for j in xrange(n):
				K[i,j] = kernels.ker(self.data_[i,0:-1], self.data_[j,0:-1], self.kernel_option_)
		
		# W, V = MDS.find_coordinates(K)
		# print "W"
		# print W
		# print "V"
		# print V
		return K

	def plot_separation(self):

		x1 = np.linspace(-3, 3, 100)
		x2 = np.linspace(-3, 3, 100)
		X1, X2 = np.meshgrid(x1, x2)
		n1, m1 = X1.shape
		n2, m2 = X2.shape
		assert(n1 == n2)
		assert(m1 == m2)

		z = np.empty([n1, m1], dtype = np.float64)
		for i in xrange(n1):
			for j in xrange(m1):
				z[i][j] = self.predict_weight(X1[i][j], X2[i][j])
		levels = [0]
		cp = plt.contour(X1, X2, z, levels, colors = 'k')


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

		y = self.data_[:, -1:]
		y_predict = self.predict()
		# print "predicted labels are:"
		# print y_predict
		for i in xrange(len(self.data_)):
			if y[i] > 0 and y_predict[i] > 0:
				predict11.append(self.data_[i, 0:-1])
			if y[i] > 0 and y_predict[i] < 0:
				predict12.append(self.data_[i, 0:-1])
			if y[i] < 0 and y_predict[i] < 0:
				predict22.append(self.data_[i, 0:-1])
			if y[i] < 0 and y_predict[i] > 0:
				predict21.append(self.data_[i, 0:-1])

		print "number of false negatives:"
		print len(predict12)
		print "number of false positives:"
		print len(predict21)
		
		for x in predict11:
			plt.plot(x[0], x[1], 'or')
		for x in predict12:
			plt.plot(x[0], x[1], 'om')
		for x in predict21:
			plt.plot(x[0], x[1], 'oc')
		for x in predict22:
			plt.plot(x[0], x[1], 'ob')



		
# http://matplotlib.org/api/pyplot_api.html





