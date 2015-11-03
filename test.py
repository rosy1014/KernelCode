import genData
import perceptron

p = perceptron.Perceptron(1)

p.generateRandomData("ga", 200)
p.choose_kernel_option(0)
p.fit()
print "the weight is: "
print p.weight_
print p.kernel_option_
p.test_classifier()