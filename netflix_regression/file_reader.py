import numpy as np 
import datetime
from sklearn.decomposition import PCA
import sys
sys.path.insert(0, '..')
import machine_learning_utils

class file_reader:
        data = {}
        def __init__(self, movie_features, train_file, test_file):
                self.data['movies'] = np.genfromtxt(movie_features, delimiter = ',', dtype = float)[1:]
                self.data['best state'], self.data['correlation coefficients'] = self.read_movie_features(movie_features)
                self.data['movies'] = self.non_linear_transformation()
                self.data['test ratings']   = np.genfromtxt(test_file, delimiter = ',', dtype = int)[1:]
                self.data['train ratings']  = np.genfromtxt(train_file, delimiter=',', dtype = int)[1:]
                self.data['movies'][:, 0] = 1
                self.data['train features'] = self.data['movies'][self.data['train ratings'][:, 1] - 1]
                self.data['test features'] = self.data['movies'][self.data['test ratings'][:, 1] - 1]


        def calculate_correlation_coeff(self, labels):
                data = self.data['movies']
                best_state = []
                pearsonCoefficients = []
                featureDimension = len(data[0])
                minimum = -0.0001
                x = data[:, 1:featureDimension]
                x_mean = np.mean(x, keepdims = True, axis = 0)
                x2_mean = np.mean(x**2, keepdims = True, axis = 0)
                for i in range(1, featureDimension):
                        y = data[:, (i+1):featureDimension]
                        y_mean = x_mean[:, i:featureDimension - 1]
                        for j in range(i, featureDimension - 1):
                                y_mean = x_mean[:, j]
                                pearsonCoefficient = np.mean(x[:, (i - 1)] * y[:, (j - i)]) - x_mean[:, (i - 1)] * y_mean/np.sqrt((x2_mean[:, (i-1)] - (x_mean[:, (i-1)]**2)) * (np.mean(y[:, (j - i)]**2) - (y_mean * y_mean)))
                                if pearsonCoefficient < minimum:
                                        best_state = [labels[i], labels[j+1], pearsonCoefficient]
                                        minimum = pearsonCoefficient
                                pearsonCoefficients.append(pearsonCoefficient)
                return best_state, pearsonCoefficients


        def non_linear_transformation(self):
                featureDimension = len(self.data['movies'][0])
                temp = np.array([[0] * 172] * len(self.data['movies']))
                temp[:, 0:featureDimension] = self.data['movies'][:, 0:featureDimension]
                curr = featureDimension
                for i in range(1, featureDimension):
                        x = self.data['movies'][:, i]
                        for j in range(i+1, featureDimension):
                                y = self.data['movies'][:, j]
                                temp[:, curr] = x * y
                                curr = curr + 1
                return temp

        def read_movie_features(self, movieFile):
                with open(movieFile) as f:
                        labels = f.readline()
                        labels = labels.split()
                        start_time = datetime.datetime.now()
                        best_state, pearsonCoefficients = self.calculate_correlation_coeff(labels)
                        end_time = datetime.datetime.now()
                        return best_state, pearsonCoefficients

        def fetch_data(self, pre_process_fn=machine_learning_utils.z_score):
            self.data['train features'][:, 1:], self.data['test features'][:, 1:] = pre_process_fn(self.data['train features'][:, 1:], self.data['test features'][:, 1:])
            return self.data
