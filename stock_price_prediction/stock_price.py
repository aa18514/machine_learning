import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.svm as svm
from functools import reduce
import os
import locale
from locale import atof
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
import sklearn.linear_model as d
from sklearn.model_selection import GridSearchCV
import sys
sys.path.insert(0, '..')
import machine_learning_utils

DFS = list()

def train_mlp_regressor(X_train, Y_train, X_test):
    print("this is X shape: {}".format(X_train.shape))
    model = Sequential()
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(lr=0.001),
                  metrics=['mse'])
    model.fit(X_train, Y_train, epochs=900, batch_size=128)
    return model.predict(X_test)


def extract_data(header: str, idx)->str:
    data = pd.read_csv('stocks\\' + header + 'data.csv', index_col = False)
    data = data.drop('Name', axis=1)
    data.date = pd.DatetimeIndex(data.date.values)
    data.index = data.date
    data = data.reindex(idx, fill_value=0)
    data.date = data.index
    data.columns.values[1:] = header + data.columns.values[1:]
    data = data.fillna(0)
    DFS.append(data)

def pre_process_data(stock_data, sp_data):
    stock_data = stock_data.drop('date', axis=1)
    stock_data = stock_data[:-1]
    sp_data = sp_data[1:]
    sp_data['Price'] = sp_data['Price'].str.replace(",", "").astype(float)
    data_sets = np.hstack([stock_data.values, sp_data['Price'].values.reshape(stock_data.shape[0], 1)])
    data_sets = data_sets[::-1]
    return data_sets

def create_train_test_patches(data_sets, alpha=0.85):
    n = int(np.floor(alpha * data_sets.shape[0]))
    X_train = data_sets[:n][:,:-1]
    Y_train = data_sets[:n][:,-1]
    X_test = data_sets[n:][:,:-1]
    Y_test = data_sets[n:][:,-1]
    return (X_train, Y_train), (X_test, Y_test)

def svm_regressor(X_train, Y_train, X_test):
    parameters = {
                'kernel': ('linear', 'rbf','poly'),
                'C':[1.5, 10],
                'gamma': [1e-7, 1e-4],
                'epsilon':[0.1,0.2,0.5,0.3]
            }
    svr = svm.SVR()
    clf = GridSearchCV(svr, parameters)
    Y_pred = clf.fit(X_train, Y_train).predict(X_test)
    print(clf.best_params_)
    return Y_pred

if __name__ == "__main__":
    fileNames = os.listdir('stocks')
    headers = []
    dates = pd.read_csv('stocks\\AAPL_data.csv')
    idx = dates['date'].unique()
    idx = pd.to_datetime(idx, format="%Y-%m-%d")
    for value in fileNames:
        if value != 'S&P 500 Historical Data.csv':
            headers.append(value[:value.find('_')+1] + 'open')
            extract_data(value[:value.find('_')+1], idx)
    stock_data = reduce(lambda left,right: pd.merge(left,right,on='date'), DFS)
    sp_data = pd.read_csv('stocks\\S&P 500 Historical Data.csv')
    data_sets = pre_process_data(stock_data, sp_data)
    (X_train, Y_train), (X_test, Y_test) = create_train_test_patches(data_sets)
    X_train, X_test = machine_learning_utils.z_score(X_train, X_test)
    r = lm.LinearRegression()
    a = np.logspace(-5, 3, 100)
    h = d.RidgeCV(alphas=a, cv=None)
    #Y_pred = train_mlp_regressor(X_train, Y_train, X_test)
    Y_pred = svm_regressor(X_train, Y_train, X_test)
    plt.ylabel('S&P price')
    plt.xlabel('time steps')
    plt.plot(Y_test, label='S&P actual stock price')
    plt.plot(Y_pred, label='S&P predicted stock price')
    plt.legend()
    plt.show()
