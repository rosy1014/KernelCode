import genData
import perceptron
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lg
from mpl_toolkits.mplot3d import Axes3D


def gaussian(x, y, sigma):
	""" x,y are numpy vectors
	"""
	gaussian = np.exp(-lg.norm(x-y)**2/(2 * (sigma ** 2)))
	return gaussian


def gaussian_grid_matrix(ngrids, sigma):
	# pairwise formular: 
	# np.exp(-lg.norm(x-y)**2/(2 * (sigma ** 2)))
	# sigma = 1
	p = perceptron.Perceptron(0.02)

	# n = math.sqrt(npoints)
	x1 = np.linspace(-1, 1, ngrids)
	x2 = np.linspace(-1, 1, ngrids)
	X, Y = np.meshgrid(x1,x2)

	X_list = np.concatenate(X)
	Y_list = np.concatenate(Y)

	XX = np.vstack([X_list, Y_list])

	LXX = X_list ** 2 + Y_list ** 2

	DXX = np.dot(XX.T, XX)

	RLXX = np.tile(LXX, (X_list.shape[0], 1))

	D = RLXX + RLXX.T - 2 * DXX
	K = np.exp(-D/(2 * sigma ** 2))
	return K


K = gaussian_grid_matrix(30, 1)

def find_coordinates_kernel(K):
	n = K.shape[0]
	H = np.identity(n) - 1.0/n * np.ones(K.shape)
	B = np.dot(H,np.dot(K,H))

	W, V = lg.eigh(B)
	idx = np.argsort(W)[::-1]
	W = W[idx]
	V = V[:,idx]
	V = V.T
	L = np.diag(np.sqrt(np.maximum(W,0)))
	Y = np.dot(L,V)
	return W, Y

W, Y = find_coordinates_kernel(K)
print Y.shape

fig = plt.figure()
# ax = fig.add_subplot(211, projection='3d', aspect='equal')
# ax.plot(Y[0], Y[1], Y[2], ".-")

as2 = fig.add_subplot(111, projection='3d', aspect = 'equal')
as2.plot(Y[0,:], Y[1,:], Y[2,:], ".")
axis_lim = np.amax(abs(Y)) * 1.2
as2.set_xlim([-axis_lim, axis_lim])
as2.set_ylim([-axis_lim, axis_lim])
as2.set_zlim([-axis_lim, axis_lim])

plt.show()