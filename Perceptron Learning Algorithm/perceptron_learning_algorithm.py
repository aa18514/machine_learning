import matplotlib.pyplot as plt
import random
import numpy as np
a = 0.3
b = 0.2
c = 0.5


def plot_linear_classifier(w):
	"""w is the updated weight vector"""
	t = np.arange(-90, 90, 0.1)
	if w[2] == 0 and w[1] != 0: 
		plt.axvline(x = (-w[0]/w[1]))
	else:
		print(w)
		plt.plot(t, (-(w[0] + w[1]*t)/w[2]))

def plot_data_points(inputs, y, sample_size):
	"""plot the 2-D features on the x and the y axis, a point is '+' if corresponding y value is +1, otherwise it is '-1'"""
	for i in range(0, sample_size): 
		if y[i] == 1:
			plt.plot(inputs[i][1], inputs[i][2], 'r+', markersize = np.sqrt(20))
		else:
			plt.plot(inputs[i][1], inputs[i][2], 'yo', markersize = np.sqrt(35))


def perceptron(inputs, y, sample_size):
	"""weight vector has been initialized to zero, keep updating the weight vector until sign(w**T . x(i)) = y(i)"""
	w = np.array([0, 0, 0])
	i = 1 
	while i < sample_size+1: 
		if np.dot(inputs[i-1], w)*y[i-1] <= 0: 
			z = inputs[i-1]*y[i-1]
			w = np.add(z, w)
			i = 1 
		else:
			i = i + 1 
	return w

def generate_dataset(sample_size): 
	"""in order to make sure that the date is linearly seperable I use an equation of the form ax + by + c = 0"""
	inputs = np.array([[1, 0, 0] for i in range(sample_size)])
	y = np.array([0]*sample_size)
	for i in range(0, sample_size): 
		inputs[i][1] = random.uniform(-sample_size,sample_size)
		inputs[i][2] = random.uniform(-sample_size,sample_size)
		if a * inputs[i][1] + b * inputs[i][2] + c <= 0: 
			y[i] = 1
		else:
			y[i] = -1
	return inputs, y

def part_a():
	fig = plt.figure()
	fig.suptitle('Perceptron Learning Algorithm')
	plt.xlabel('x1')
	plt.ylabel('x2')
	sample_size = [2, 4, 10, 100]
	for i in range(0, 4):
		fig = plt.figure(i+1)
		fig.suptitle("Q3 a) with sample size %s" % sample_size[i])
		inputs, y = generate_dataset(sample_size[i])
		plot_data_points(inputs, y, sample_size[i])
		w = perceptron(inputs, y, sample_size[i]);
		w = perceptron(inputs, y, sample_size[i]) 
		plot_linear_classifier(w);
		plt.xlim(-120, 120)
		plt.ylim(-120, 120)
		plt.xlabel('x1')
		plt.ylabel('x2')
	plt.show()
