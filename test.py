import genData
import perceptron
import matplotlib.pyplot as plt

p = perceptron.Perceptron(0.02)

p.generateRandomData("ga", 100)
p.choose_kernel_option(0)
p.fit()
p.plot_separation()
print "the weight is: "
print p.weight_
# print p.kernel_option_
p.test_classifier()
plt.show()