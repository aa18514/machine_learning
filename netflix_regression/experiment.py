"""cross validated scores"""
from sklearn.linear_model import Lars
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import lasso_path
import numpy as np
import matplotlib.pyplot as plt
import pickle
from itertools import cycle

def main():
    feature_vectors = []
    movie_features = []
    features_train = []
    data = None
    with open("movie-data/ratings-train.csv") as f:
        data = f.readlines()
    data = data[1:]
    for val in data:
        line = val[:-1]
        line = line.split(",")
        line = [float(val) for val in line]
        feature_vectors.append(line)
    feature_vectors = np.array(feature_vectors)
    ratings_train = feature_vectors[:, 2]
    movie_ids = feature_vectors[:, 1]
    data = None
    with open("movie-data/movie-features.csv") as f:
        data = f.readlines()
    data = data[1:]
    for val in data:
        line = val[:-1]
        line = line.split(",")
        line = [float(val) for val in line]
        movie_features.append(line)
    movie_features = np.array(movie_features)
    error = []
    error_lars = []
    error_lasso = []
    for i in range(671):
        person_features = feature_vectors[(feature_vectors[:,0] - 1) == i]
        for k in range(len(person_features[:, 2])):
            person_features[:,2][k] = (person_features[:,2][k] - np.mean(person_features[:, 2]))/(np.std(person_features[:, 2]))
        MOVIE_IDS = person_features[:, 1]
        features_train = movie_features[np.array(MOVIE_IDS, int) - 1]
        features_train[:, 0] = 1.0
        for p in range(1, features_train.shape[1]):
            features_train[:, p] = (features_train[:, p] - np.mean(features_train[:, p]))/(np.std(features_train[:, p]) + 10**-8)
        lasso = Lasso(alpha = 0.01, normalize=True, fit_intercept = True)
        alphas, coeff, _ = lasso_path(features_train, person_features[:, 2], 5e-3, positive = True, fit_intercept = False)
        alphas = -np.log10(alphas)
        colors = cycle(['b', 'r', 'g', 'c', 'k'])
        plt.xlabel('alphas')
        plt.ylabel('coeff')
        k = 0
        for val, c in zip(coeff, colors):
            print(k)
            k = k + 1
            print(np.array(val).shape)
            print(np.array(alphas).shape)
            plt.plot(alphas, val, c=c)
        plt.show()
        pred = lasso.fit(features_train, person_features[:, 2]).predict(features_train)
        error_lasso.append(np.mean((pred - person_features[:, 2])**2))
        clf = Ridge(alpha = 0.1, normalize=True, fit_intercept = True)
        reg = Lars(n_nonzero_coefs = 5)
        pred = clf.fit(features_train, person_features[:, 2]).predict(features_train)
        error.append(np.mean((pred - person_features[:, 2])**2))
        pred = reg.fit(features_train, person_features[:, 2]).predict(features_train)
        #pred[pred < 0] = 0
        error_lars.append(np.mean((pred - person_features[:, 2])**2))
        print(np.mean((pred - person_features[:, 2])**2))
	#print(features_train)
	#print(person_features)
    plt.figure(1)
    #plt.title('least angles regression')
    plt.xlabel('users')
    plt.ylabel('error')
    line_up, = plt.plot(error_lars, label='least angles regression')
    #plt.figure(2)
    #plt.xlabel('users')
    #plt.ylabel('error')
    #plt.title('ridge regression')
    line_down, = plt.plot(error, label = 'ridge regression')
    #plt.figure(3)
    #plt.xlabel('users')
    #plt.ylabel('error')
    #plt.title('lasso regression')
    line_hoz, = plt.plot(error_lasso, label = 'lasso regression')
    plt.legend(handles=[line_up, line_down, line_hoz])
    plt.show()
    print("lar error: " + str(np.mean(error_lars)))
    print("ridge error: " + str(np.mean(error)))
    print("lasso error: " + str(np.mean(error_lasso)))
    #print(movie_ids)
    #print(np.array(movie_features[:, 0], int) == movie_ids)
    features_train = movie_features[np.array(movie_ids, int) - 1]
    features_train[:, 0] = 1.0
    #for i in range(1, features_train.shape[1]):
    #    features_train[:, i] = (features_train[:, i] - np.mean(features_train[:, i]))/(np.std(features_train[:, i]))
    features_train = (features_train - np.mean(features_train))/np.std(features_train)
    clf = Ridge(alpha = 0.1, normalize=True, fit_intercept = True)
    pred = clf.fit(features_train, ratings_train).predict(features_train)
    reg = Lars(n_nonzero_coefs = np.inf)
    #pred = reg.fit(features_train, ratings_train).predict(features_train)
    error = (pred - ratings_train)**2
    plt.plot(error)
    plt.show()
    #print(error)
    #print(reg.coef_.shape)
    #print(features_train.shape)
    #print(features_train)
        #feature_vectors.append(val[:-1]
    #print(data)


if __name__ == "__main__":
    main()
