import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
import argparse
import textwrap
import datetime
import itertools
from sklearn import linear_model
from itertools import chain
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
from file_reader import compute_pca
from datetime import timedelta
from file_reader import file_reader
from joblib import Parallel, delayed
import multiprocessing


def quantize(expected_ratings):
    """
    quantize movie ratings nearest to 0.5, any ratings less than
    0 are capped at 0, and any ratings beyond 5.0 are capped at 5.0.
    @expected_ratings: predicted ratings obtained using regression
    """
    ratings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.50]
    expected_ratings[np.where(expected_ratings < 0.25)] = 0.00
    for val in ratings:
        expected_ratings[np.where(np.logical_and(expected_ratings >= val - 0.25,\
                                                 expected_ratings <= val))] = val
        expected_ratings[np.where(np.logical_and(expected_ratings > val,\
                                                 expected_ratings < val + 0.25))] = val
    expected_ratings[np.where(expected_ratings >= 4.75)] = 5.00
    return expected_ratings


def compute_error(weights, test_ratings, movie_features):
    """
    return MSE between actual and predicted ratings
    weights: weights corresponding to the movie features
    movie features: 1/0 indicating if the paticular genre exists
    """
    expected_ratings = quantize(np.dot(weights, np.array(movie_features).T))
    return np.mean((expected_ratings - test_ratings)**2)


def train_dataset(feature_dimension, features, ratings, lambd_val):
    """
    train ridge regressor where ratings is the predicted values
    and features indicate the feature vector
    lambd val: L2 penalty (used to control overfitting
    """
    clf = linear_model.Ridge(alpha=lambd_val, normalize=False, fit_intercept=False, solver='lsqr')
    clf.fit(features.reshape(len(ratings), feature_dimension), ratings)
    return clf.coef_


def k_fold_algorithm(movie_ratings, feature_dimension, res, values_of_lambda, K):
    """
    run cross fold validation corresponding to the value of regularization parameter and
    different values for K, i.e the number of folds
    @use multiprocessing to parallelize the cross fold validation
    """
    errors = []
    cv = KFold(len(movie_ratings), n_folds=K)
    for obj in cv:
        error = []
        for value in values_of_lambda:
            weight = train_dataset(feature_dimension, movie_ratings[obj[0]], res[obj[0]], value)
            error.append(compute_error(weight, res[obj[1]], movie_ratings[obj[1]]))
        errors.append(error)
    errors = np.sum(errors, axis=0, keepdims=True)
    reg_constant = values_of_lambda[np.argmin(errors)]
    return reg_constant, train_dataset(feature_dimension, movie_ratings, res, reg_constant)


def naive_linear_regression(movie_ratings, res):
    """
    a variant of ridge regressor where lambda = 0
    @movie_ratings: 19 dimensional feature vectors
    @res: actual value of the rating
    """
    return np.dot(np.linalg.pinv(movie_ratings), res.T)


def compute(weight, partitioned_test_ratings, partitioned_movie_features):
    errors = np.array([])
    for users in range(671):
        e = quantize(np.dot(weight[users], partitioned_movie_features[users].T))
        errors = np.append(errors, np.mean((e - partitioned_test_ratings[users])**2))
    return errors


def processInput(i, ratings, movie_features, algorithm, args):
    """
    processInput multiplexes between native linear regression, linear regression
    with non-linear transformation and ridge regressor
    the processInput function is parallelized between multiple users amongst n cores
    where n is the maximum number of cores present on the CPU
    """
    person = ratings[(ratings[:, 0] - 1) == i]
    movie_ratings = movie_features[np.where(ratings[:, 0] - 1 == i)][:, 1:]
    feature_dimension = len(movie_ratings[0])
    w = None
    rg_constant = None
    if algorithm == "lin_reg":
        w = naive_linear_regression(movie_ratings, person[:, 2])
    elif algorithm == "k_fold":
        rg_constant, w = k_fold_algorithm(movie_ratings, \
                                          feature_dimension, person[:, 2], args[0], args[1])
    return [rg_constant], movie_ratings, person[:, 2], w, [i]


def extract_person(ratings, algorithm, movie_features, *args, **kwargs):
    j = np.array(Parallel(n_jobs=multiprocessing.cpu_count())\
            (delayed(processInput)(i, ratings, movie_features, algorithm, args) for i in range(671)))
    regularized_constants = np.array([x for _, x in sorted(zip(j[:, 4], j[:, 0]))])
    weight = np.array([x for _, x in sorted(zip(j[:, 4], j[:, 3]))])
    partitioned_ratings = np.array([x for _, x in sorted(zip(j[:, 4], j[:, 2]))])
    partitioned_movie_features = np.array([x for _, x in sorted(zip(j[:, 4], j[:, 1]))])
    return regularized_constants, partitioned_ratings, partitioned_movie_features, weight


def compute_test_error(weight, test_ratings, movie_features):
    _, partitioned_test_ratings, partitioned_movie_features, _ = \
        extract_person(test_ratings, "None", movie_features)
    return compute(weight, partitioned_test_ratings, partitioned_movie_features)


def compute_train_error(train_ratings, movie_features, algorithm, *args, **kwargs):
    regularized_constants, partitioned_train_ratings, \
    partitioned_movie_features, weight = \
        extract_person(train_ratings, algorithm, movie_features, args[0], args[1])
    return regularized_constants, weight,\
           compute(weight, partitioned_train_ratings, partitioned_movie_features)


def func(k, movie_features, train_data):
    regularized_constants, weight, train_error = \
        compute_train_error(f.read_train_data(), movie_features, "k_fold", np.logspace(-4, 0, 50), k)
    return regularized_constants, train_error, weight


def plot_data(title, xlabel, ylabel, x, y, *args, **kwargs):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if not args:
        plt.plot(x, y)
    else:
        plt.plot(x, y, args[0])
    plt.show()


def compute_multiple_pca(i, TrainRatings, TestRatings, TrainMovieFeatures, TestMovieFeatures, n_features, epsilon=10**-8):
    person_train = TrainRatings[TrainRatings[:, 0] == i]
    person_test = TestRatings[TestRatings[:, 0] == i]
    train_movie_ratings = TrainMovieFeatures[person_train[:, 1] - 1]
    test_movie_ratings = TestMovieFeatures[person_test[:, 1] - 1]
    mean = np.mean(train_movie_ratings[:, 1:], axis=0)
    std = np.std(train_movie_ratings[:, 1:], axis=0)
    train_movie_ratings[:, 1:] = (train_movie_ratings[:, 1:] - mean)/(std + epsilon)
    test_movie_ratings[:, 1:] = (test_movie_ratings[:, 1:] - mean)/(std + epsilon)
    pca = PCA(n_components=n_features)
    pca.fit(train_movie_ratings)
    a = pca.transform(train_movie_ratings)
    return [1], a, pca.transform(test_movie_ratings)


def standardize(data, mean, std, epsilon=10**-8):
    return (data - mean)/(std + epsilon)

def linear_regression_with_regularization(movie_features, train_ratings, test_ratings, n_features, args, epsilon=10**-8):
    """a total of 671 users, 700003 movies, the function handles linear_model regression
    for each user, linear regression with regularization, and non -linear transformation """
    tr = movie_features[train_ratings[:, 1] - 1]
    ts = movie_features[test_ratings[:, 1] - 1]
    tr[:, 1:] = standardize(tr[:, 1;], np.mean(tr[:, 1], axis = 0), \
                            np.std(tr[:, 1], axis = 0))
    
    ts[:, 1:] = standardize(ts[:, 1:], np.mean(tr[:, 1], axis = 0), \
                            np.std(tr[:, 1], axis = 0))
    
    pca = PCA(n_components=n_features)
    pca.fit(tr)
    b = np.ones((70002, n_features + 1))
    c = np.ones((30002, n_features + 1))
    b[:, 1:] = pca.transform(tr)
    c[:, 1:] = pca.transform(movie_features[test_ratings[:, 1] - 1])
    b[:, 0] = train_ratings[:, 1]
    c[:, 0] = test_ratings[:, 1]

    if args.verbose == 1 or args.verbose == 3:
        K = [3, 4, 5, 6, 7, 8]
        regularized_constants = []
        train_errors = []
        final_weights = []
        for i in K:
            rg, train_error, weight = func(i, b, train_ratings)
            regularized_constants.append(rg)
            train_errors.append(train_error)
            final_weights.append(weight)
        regularized_constants = np.array(regularized_constants)
        train_errors = np.array(train_errors)
        final_weights = np.array(final_weights)
        error = np.mean(train_errors, axis=1) + np.var(train_errors, axis=1)
        minimum = np.argmin(error)
        return train_errors[minimum], compute_test_error(final_weights[minimum], test_ratings, c)
    else:
        _, weight, train_error = compute_train_error(train_ratings, b, "lin_reg", None, None)
        return train_error, compute_test_error(weight, test_ratings, c)


def exponential_weightings(error, beta):
    vo = 0.0
    error = (1 - beta) * error
    vs = []
    for err in range(len(error)):
        vo = beta * vo + err
        vs.append(vo)
    return vs


def regression_analysis(movie_features, train_ratings, test_ratings, args):
    beta = 0.9
    times = []
    test_bias = []
    errors = []
    train_errors = []
    train_variance = []
    test_variance = []
    number_of_features = 50

    for i in range(1, number_of_features):
        start_time = datetime.datetime.now()
        error_train, error_test = linear_regression_with_regularization(movie_features, train_ratings, test_ratings, i, args)
        finish_time = datetime.datetime.now()
        times.append((finish_time - start_time).total_seconds())
        errors.append(np.mean(error_test))
        train_errors.append(np.mean(error_train))
        test_variance.append(np.var(error_test))
        train_variance.append(np.var(error_train))

    p = np.poly1d(np.polyfit(np.arange(1, number_of_features, 1), times, 3))
    x = np.arange(1, number_of_features, 1)
    plt.plot(x, p(x))
    plt.plot(x, times, 'r+')
    plt.title('PCA analysis')
    plt.xlabel('n_components')
    plt.ylabel('time/s')
    plt.show()
    fig = plt.figure()
    plt.title('PCA analysis')
    ax = plt.subplot(111)
    ax.plot(np.arange(1, number_of_features, 1), errors, label='test bias')
    ax.plot(np.arange(1, number_of_features, 1), train_errors, label='train error')
    ax.legend()
    #plt.legend((ax_train, ax_test), ('train bias', 'test bias'))
    plt.xlabel('n_components')
    plt.ylabel('bias')
    plt.show()
    plt.title('variance')
    ax = plt.subplot(111)
    ax.plot(np.arange(1, number_of_features, 1), train_variance, label='test variance')
    ax.plot(np.arange(1, number_of_features, 1), test_variance, label='train variance')
    ax.legend()
    #plt.legend((ax_train, ax_test), ('train bias', 'test bias'))
    plt.xlabel('n_components')
    plt.ylabel('variance')
    plt.show()
    #print("program took: %f s" % ((b-a).total_seconds()))
    #print("train bias: %f"  % (np.mean(error_train)))
    #print("train var:  %f"  % (np.var(error_train)))
    #print("test bias:  %f"  % (np.mean(error_test)))
    #print("test var:   %f"  % (np.var(error_test)))


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent
        ('''\
                    netflix dataset for 671 users and 9066 movies
                    all data is stored in csv files in the sub-directory /movie-data
                    all expected ratings are quantised to the nearest 0.5
                    use -v to enable linear regression with cross fold validation
                    use --v to enable naive linear regression
        ''')

    )
    PARSER.add_argument('-v', '--verbose', action="count", help="used to switch between linear regression with and w/o cross_validation")
    ARGS = PARSER.parse_args()
    FILE = file_reader("movie-data\\movie-features.csv", "movie-data\\ratings-train.csv", "movie-data\\ratings-test.csv")
    BEST_STATE, PEARSON_COEFFICIENTS, MOVIE_FEATURES = FILE.read_movie_features(ARGS)
    if ARGS.verbose != 0:
        regression_analysis(MOVIE_FEATURES, FILE.read_train_data(), FILE.read_test_data(), ARGS)
        print("pearson coefficient between %s and %s is %f" % (BEST_STATE[0], BEST_STATE[1], BEST_STATE[2]))
        plt.plot(PEARSON_COEFFICIENTS, 'g*')
        plt.xlabel('genre tuple')
        plt.ylabel('correlation coefficient')
        plt.show()
