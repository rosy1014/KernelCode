import genData
import matplotlib.pyplot as plt
# import numpy as np
from matplotlib import animation
import time
import linear_perceptron 
import perceptron 
# import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lg
from mpl_toolkits.mplot3d import Axes3D
import math

p = perceptron.Perceptron(0.02)
rnd = p.generateRandomData("ga", 10)
p.choose_kernel_option(1)
p.fit()

p2 = linear_perceptron.Perceptron(0.02)
print rnd[:,0]
print rnd[:,1]
phi_data_z = rnd[:,0]**2 + rnd[:,1]**2 
phi_data_w = np.ones(rnd.shape[0])
phi_data = np.column_stack((rnd[:,0], rnd[:,1], phi_data_z, phi_data_w, rnd[:,2]))
p2.data_ = phi_data
print phi_data
p2.fit()
print p2.weight_
xn = (np.random.rand(1)*2 - 1)/2 + 0.5
yn = (np.random.rand(1)*2 - 1)/2 + 1 

f1 = plt.figure()
f2 = plt.figure()
f3 = plt.figure()
f4 = plt.figure()
f5 = plt.figure()
f6 = plt.figure()
f7 = plt.figure()



rnd_plot = f1.add_subplot(111, aspect='equal')
for x in rnd:
	if x[-1]>0:
		rnd_plot.plot([x[0]],[x[1]],c='b', marker = 'o')
	else:
		rnd_plot.plot([x[0]],[x[1]], c = 'r', marker = 'o')
# rnd_plot.plot(xn,yn, '*m', markersize=20)

black_plot = f2.add_subplot(111,projection = '3d', aspect='equal',sharex = rnd_plot)
for x in rnd:
	if x[-1]>0:
		black_plot.plot([x[0]],[x[1]],zs=x[0]**2 + x[1]**2 , zdir='z', label='zs=0, zdir=z',c='b', marker = 'o')
	else:
		black_plot.plot([x[0]],[x[1]],zs=x[0]**2 + x[1]**2 , zdir='z', label='zs=0, zdir=z',c='r', marker = 'o')


rnd_plot2 = f3.add_subplot(111,aspect='equal', sharex = rnd_plot)
for x in rnd:
	if x[-1]>0:
		rnd_plot2.plot(x[0],x[1],'sb')
	else:
		rnd_plot2.plot(x[0],x[1],'or')

twou_plot = f4.add_subplot(111,aspect='equal',sharex = rnd_plot)
for x in rnd:
	if x[-1]>0:
		twou_plot.plot(x[0],x[1],'sb')
	else:
		twou_plot.plot(x[0],x[1],'or')
# twou_plot.plot(xn,yn,'*c', markersize = 20)
p.plot_separation(twou_plot)

d3sep = f5.add_subplot(111, projection='3d', aspect='equal', sharex = rnd_plot, sharez = black_plot)
for x in rnd:
	if x[-1]>0:
		d3sep.plot([x[0]],[x[1]],zs=x[0]**2 + x[1]**2, zdir='z', label='zs=0, zdir=z',c='b', marker = 'o')
	else:
		d3sep.plot([x[0]],[x[1]],zs=x[0]**2 + x[1]**2 , zdir='z', label='zs=0, zdir=z',c='r', marker = 'o')

rnd_plot.set_xlim([-3, 3])
rnd_plot.set_ylim([-3.5, 3.5])

emp = f6.add_subplot(111, aspect='equal', sharex = rnd_plot)
emp.grid(True)
emp.set_xlim([-3,3])
emp.set_ylim([-3.5,3.5])

emp2 = f7.add_subplot(111, projection = '3d', aspect='equal', sharex = rnd_plot, sharez = black_plot)
emp2.grid(True)
emp2.set_xlim([-3,3])
emp2.set_ylim([-3.5,3.5])
# rnd_plot.set_zlim([0,10])
plt.setp(rnd_plot.get_xticklabels(),  visible = False)
plt.setp(black_plot.get_xticklabels(),  visible = False)
plt.setp(rnd_plot2.get_xticklabels(),  visible = False)
plt.setp(twou_plot.get_xticklabels(),  visible = False)
plt.setp(rnd_plot.get_yticklabels(),  visible = False)
plt.setp(black_plot.get_yticklabels(),  visible = False)
plt.setp(rnd_plot2.get_yticklabels(),  visible = False)
plt.setp(twou_plot.get_yticklabels(),  visible = False)

# plt.setp(rnd_plot.get_zticklabels(), visible = False)
plt.setp(black_plot.get_zticklabels(), visible = False)
plt.setp(d3sep.get_zticklabels(), visible = False)
plt.setp(d3sep.get_yticklabels(), visible = False)
plt.setp(d3sep.get_xticklabels(), visible = False)


plt.setp(emp.get_yticklabels(), visible = False)
plt.setp(emp.get_xticklabels(), visible = False)
plt.setp(emp2.get_zticklabels(), visible = False)
plt.setp(emp2.get_yticklabels(), visible = False)
plt.setp(emp2.get_xticklabels(), visible = False)
# f.tight_layout()
plt.show()