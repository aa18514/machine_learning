import numpy as np
import matplotlib.pyplot as plt 
import scipy 
from scipy import stats
from sklearn import linear_model
from sklearn import decomposition
import math

def read_from_file(filename):
	res = []  
	with open(filename, "r") as csvreader: 
		csvreader.readline()
		result = csvreader.readlines()
		for i in range(0, len(result)): 
			line = result[i].strip('\n')
			temp = line.split(",")
			res.append([float(i) for i in temp])
	return res

def part_d(train_ratings, K): 
	learning_rate = 0.0001
	features_matrix = np.random.rand(K, 9066)
	users_matrix =  np.random.rand(K, 671) 
	for epochs in range(0, 100):
		for i in range(0, 70002):  
			error = train_ratings[i][2] - np.dot(np.transpose(users_matrix[:,int(train_ratings[i][0] - 1)]), features_matrix[:,int(train_ratings[i][1]-1)])
			temp_users = users_matrix[:, int(train_ratings[i][0] -1)]
			users_matrix[:,int(train_ratings[i][0] - 1)] = users_matrix[:, int(train_ratings[i][0] - 1)] + learning_rate * error * features_matrix[:,int(train_ratings[i][1]- 1)]
			features_matrix[:,int(train_ratings[i][1] - 1)] = features_matrix[:, int(train_ratings[i][1] - 1)] + learning_rate * error * temp_users
	return np.asmatrix((users_matrix).T*np.asmatrix(features_matrix))


if __name__ == "__main__": 
	test_ratings   = read_from_file("movie-data\\ratings-test.csv")
	train_ratings  = read_from_file("movie-data\\ratings-train.csv")
	regularization_constants = np.logspace(-4, 0, 50)
	user_test_error = [-1]*9066
	movie_test_error = [-1]*9066
	predicted_ratings = part_d(train_ratings, 100).tolist()

	user_error = [0]*671
	for i in range(0, 671): 
		average_user_mean = []
		for j in range(0, 30002): 
			if(int(test_ratings[j][0] - 1.0) == i): 
				average_user_mean.append((test_ratings[j][2] - predicted_ratings[i][int(test_ratings[j][1] - 1.0)])**2)
		user_error[i] = np.mean(average_user_mean)
	print("model accuracy: %f" % (100 - np.mean(user_error)))