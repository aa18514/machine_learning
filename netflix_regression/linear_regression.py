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
from sklearn.cross_validation import KFold
from datetime import timedelta
from file_reader import *
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd

DATA_SET = {}


def quantize(expected_ratings):
    """
    quantize movie ratings nearest to 0.5, any ratings less than
    0 are capped at 0, and any ratings beyond 5.0 are capped at 5.0.
    @expected_ratings: predicted ratings obtained using regression
    """
    ratings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.50]
    expected_ratings[np.where(expected_ratings < 0.25)] = 0.00
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
    pred = []
    for users in range(671):
        e = quantize(np.dot(weight[users], partitioned_movie_features[users].T))
        pred.extend(e)
        errors = np.append(errors, np.mean((e - partitioned_test_ratings[users])**2))
    return pred, errors


def processInput(i, ratings, movie_features, algorithm, args):
    """
    processInput multiplexes between native linear regression, linear regression
    with non-linear transformation and ridge regressor
    the processInput function is parallelized between multiple users amongst n cores
    where n is the maximum number of cores present on the CPU
    """
    person_id = ratings[:, 0] - 1
    person = ratings[person_id == i]
    movie_ratings = movie_features[person_id == i]
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
    j = np.array(Parallel(n_jobs=7)\
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


def compute_train_error(movie_features, algorithm, *args, **kwargs):
    train_ratings = DATA_SET['train ratings']
    regularized_constants, partitioned_train_ratings, \
    partitioned_movie_features, weight = \
        extract_person(train_ratings, algorithm, movie_features, args[0], args[1])
    pred, errors = compute(weight, partitioned_train_ratings, partitioned_movie_features)
    return regularized_constants, weight, pred, errors

def func(k, movie_features, train_data):
    regularized_constants, weight, pred, train_error = \
        compute_train_error(movie_features, "k_fold", np.logspace(-4, 0, 50), k)
    return regularized_constants, train_error, weight


def linear_regression_with_regularization(n_features, args):
    """a total of 671 users, 700003 movies, the function handles linear_model regression
    for each user, linear regression with regularization, and non -linear transformation """
    train_ratings = DATA_SET['train ratings']
    test_ratings = DATA_SET['test ratings']
    tr = DATA_SET['train features']
    ts = DATA_SET['test features']
     #b, c = pca_tansformation(tr, ts, n_features)

    if args.verbose == 1 or args.verbose == 3:
        K = [3, 4, 5, 6, 7, 8]
        regularized_constants = []
        train_errors = []
        final_weights = []
        for i in K:
            rg, train_error, weight = func(i, tr, train_ratings)
            regularized_constants.append(rg)
            train_errors.append(train_error)
            final_weights.append(weight)
        regularized_constants = np.array(regularized_constants)
        train_errors = np.array(train_errors)
        final_weights = np.array(final_weights)
        error = np.mean(train_errors, axis=1) + np.var(train_errors, axis=1)
        minimum = np.argmin(error)
        return train_errors[minimum], compute_test_error(final_weights[minimum], test_ratings, ts)
    else:
        _, weight, _, train_error = compute_train_error(tr, "lin_reg", None, None)
        a, b = compute_test_error(weight, test_ratings, ts)
        return train_error, a, b

def pca_analysis():
    times = list()
    test_bias = list()
    errors = list()
    train_errors = list()
    test_variance = list()
    number_of_features = 2
    for i in range(1, number_of_features):
        start_time = datetime.datetime.now()
        error_train, p, error_test = linear_regression_with_regularization(0, args)
        finish_time = datetime.datetime.now()
        times.append((finish_time - start_time).total_seconds())
        errors.append(np.mean(error_test))
        train_errors.append(np.mean(error_train))
        test_variance.append(np.var(error_test))
        train_variance.append(np.var(error_train))
    x = np.arange(1, number_of_features, 1)
    p = np.poly1d(np.polyfit(x, times, 3))
    plt.plot(x, p(x))
    plt.plot(x, times, 'r+')
    plt.title('PCA analysis')
    plt.xlabel('n components')
    plt.ylabel('time/s')
    plt.show()
    fig = plt.figure()
    plt.title('PCA analysis')
    ax = plt.subplot(111)
    ax.plot(x, errors, label='test bias')
    ax.plot(x, train_errors, label='train bias')
    ax.legend()
    plt.xlabel('n components')
    plt.ylabel('bias')
    plt.show()
    plt.title('variance')
    ax = plt.subplot(111)
    ax.plot(x, train_variance, label='train variance')
    ax.plot(x, test_variance, label='test variance')
    ax.legend()
    plt.xlabel('n components')
    plt.ylabel('variance')
    plt.show()

def regression_analysis(args, trials=20):
    arithmetic_mean = list()
    for i in range(trials):
       start_time = datetime.datetime.now()
       error_train, p, error_test = linear_regression_with_regularization(0, args)
       finish_time = datetime.datetime.now()
       arithmetic_mean.append((finish_time - start_time).total_seconds())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    residuals = DATA_SET['test ratings'][:, 2] - p
    normalized_residuals = machine_learning_utils.get_normalized_residuals(residuals)
    machine_learning_utils.plot_sample_variances(normalized_residuals, ax1)
    test_statistic, p_value = machine_learning_utils.barlett_test(p)
    machine_learning_utils.histogram_residuals(residuals, ax2)
    print("test_statistic, p_value: {} {}".format(test_statistic, p_value))
    plt.tight_layout()
    print("program took: {} s".format(np.mean(arithmetic_mean)))
    print("train bias: {}".format(np.mean(error_train)))
    print("train var: {}".format(np.var(error_train)))
    print("test bias: {}".format(np.mean(error_test)))
    print("test var: {}".format(np.var(error_test)))
    plt.show()

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
    FILE = file_reader(
            "movie-data\\movie-features.csv",
            "movie-data\\ratings-train.csv",
            "movie-data\\ratings-test.csv"
            )
    DATA_SET = FILE.fetch_data()
    BEST_STATE, PEARSON_COEFFICIENTS = DATA_SET['best state'], DATA_SET['correlation coefficients']
    if ARGS.verbose != 0:
        regression_analysis(ARGS)
        print("pearson coefficient between %s and %s is %f" % (BEST_STATE[0], BEST_STATE[1], BEST_STATE[2]))
        plt.plot(PEARSON_COEFFICIENTS, 'g*')
        plt.xlabel('genre tuple')
        plt.ylabel('correlation coefficient')
        plt.show()
