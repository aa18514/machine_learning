import csv
import os
import locale
import sys
import arrow
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.svm as svm
from functools import reduce
from locale import atof
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout
import sklearn.linear_model as d
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor
import keras.backend as K
from sklearn.metrics import r2_score
from scipy import signal
sys.path.insert(0, '..')
import machine_learning_utils
import datetime

DFS = list()
INTRA_DAY_DATA = list()
volumes = list()
price = list()
HEADERS = []

import re
from io import StringIO
import requests
import pandas as pd
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf


def get_quote_data(symbol='iwm', data_range='100d', data_interval='1m'):
    res = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={data_range}&interval={data_interval}'.format(**locals()))
    data = res.json()
    df = None
    try:
        body = data['chart']['result'][0]
        dt = datetime.datetime
        dt = pd.Series(map(lambda x: arrow.get(x).to('EST').datetime.replace(tzinfo=None), body['timestamp']), name='dt')
        df = pd.DataFrame(body['indicators']['quote'][0], index=dt)
        dg = pd.DataFrame(body['timestamp'])
        return df.loc[:, ('open', 'high', 'low', 'close', 'volume')]
    except Exception as e:
        print(e)
        return None


def rmse_vec(pred_y, true_y):
    return K.mean((pred_y - true_y)**2)


def percent_to_float(x: str)->str:
    return float(x.strip('%'))


def create_model(neurons=1, layers=[40, 30, 25], learning_rate=0.000035, loss='mean_squared_error'):
    model = Sequential()
    for layer in layers:
        model.add(Dense(units=layer, activation='relu'))
    model.add(Dense(units=3))
    model.compile(loss=loss,
                  optimizer=Adam(lr=learning_rate),
                  metrics=['mse'])
    return model

def train_mlp_regressor(X_train, Y_train, X_test):
    callbacks = [EarlyStopping(monitor='mse', patience=8),
                 ModelCheckpoint(filepath='best_model_1.h5', monitor='val_loss', save_best_only=True)]
    #model = KerasRegressor(build_fn=create_model, epochs=100,verbose=70)
    model = create_model()
    #neurons = [5, 10, 15, 20, 25]
    #param_grid = dict(neurons=neurons)
    #grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, verbose=50, n_jobs=4)
    model.fit(X_train, Y_train, epochs=900)
    #grid_search = grid.fit(X_train, Y_train)
    #print(grid_search.best_params_)
    return model.predict(X_test), model.predict(X_train)


def extract_data(header: str, idx)->str:
    data = pd.read_csv('stocks\\' + header + 'data.csv', index_col=False)
    data = data.drop('Name', axis=1)
    func = machine_learning_utils.log_transformation
    #data['low'][1:] = func(data['low'].values)
    #data['high'][1:] = func(data['high'].values)
    #data['open'][1:] = func(data['open'].values)
    #data['close'][1:] = func(data['close'].values)
    #data = data[1:]
    data.date = pd.DatetimeIndex(data.date.values)
    data.index = data.date
    data = data.reindex(idx, fill_value=0)
    data.date = data.index
    data.columns.values[1:] = header + data.columns.values[1:]
    data = data.fillna(method='pad')
    DFS.append(data)


def pre_process_data(stock_data, company_data, keys):
    stock_data = stock_data.drop('date', axis=1)
    stock_data = stock_data[:-1]
    company_data = company_data[1:]
    data_sets = stock_data.values
    for key in keys:
        company_data[key] = company_data[key].str.replace(",", "").astype(float)
        data_sets = np.hstack([data_sets, company_data[key].values.reshape(stock_data.shape[0], 1)])
    data_sets = data_sets[::-1]
    #for i in range(1, len(keys) + 1):
        #data_sets[:, (-1 * i)][1: ] = np.log(data_sets[:, (-1 * i)][1: ]/data_sets[:, (-1 * i)][: -1])
    return data_sets


def create_train_test_patches(data_sets, keys, alpha=0.90):
    n = int(np.floor(alpha * data_sets.shape[0]))
    X_train = data_sets[:n][:, :-4]
    Y_train = np.hstack([data_sets[:n][:, -1].reshape(len(data_sets[:n][:, -1]), 1),
                         data_sets[:n][:, -2].reshape(len(data_sets[:n][:, -1]), 1),
                         data_sets[:n][:, -3].reshape(len(data_sets[:n][:, -1]), 1),
                         data_sets[:n][:, -4].reshape(len(data_sets[:n][:, -1]), 1)])
    X_test = data_sets[n:][:, :-4]
    Y_test = np.hstack([data_sets[n:][:, -1].reshape(len(data_sets[n:][:, -1]), 1),
                        data_sets[n:][:, -2].reshape(len(data_sets[n:][:, -1]), 1),
                        data_sets[n:][:, -3].reshape(len(data_sets[n:][:, -1]), 1),
                        data_sets[n:][:, -4].reshape(len(data_sets[n:][:, -1]), 1)])
    return (X_train, Y_train), (X_test, Y_test)


def svm_regressor(X_train, Y_train, X_test):
    Y_train = Y_train.reshape(len(Y_train), )
    parameters = {
        'kernel': ['rbf', 'poly'],
        'C':[100, 500],
        'gamma': [1e-4],
        'epsilon':[100, 150]
    }
    svr = svm.SVR()
    clf = GridSearchCV(svr, parameters, n_jobs=6, verbose=10)
    Y_pred = clf.fit(X_train, Y_train).predict(X_test)
    Y_train_pred = clf.fit(X_train, Y_train).predict(X_train)
    print(clf.best_params_)
    return Y_pred, Y_train_pred


def data_acquisition():
    fileNames = os.listdir('stocks')
    print(fileNames)
    headers = []
    dates = pd.read_csv('stocks\\AAPL_data.csv')
    idx = dates['date'].unique()
    idx = pd.to_datetime(idx, format="%Y-%m-%d")
    for value in fileNames:
        if value != 'S&P 500 Historical Data.csv':
            print(value)
            HEADERS.append(value[:value.find('_')])
            extract_data(value[:value.find('_')+1], idx)
    stock_data = reduce(lambda left, right: pd.merge(left, right, on='date'), DFS)
    sp_data = pd.read_csv('stocks\\S&P 500 Historical Data.csv')
    sp_data = sp_data[1:]
    return stock_data[1: ], sp_data


def extract_from_csv(write_to_csv=True):
    keys = ["Low", "High", "Open"]
    stock_data, sp_data = data_acquisition()
    data_sets = pre_process_data(stock_data, sp_data, keys)
    indices = list()
    for i in range(len(keys) - 1, 0):
        indices.append(sp_data[keys[i]].str.replace(",", "").astype(float).values)
    #k = k[:-1]
    (X_train, Y_train), (X_test, Y_test) = create_train_test_patches(data_sets, keys)
    if write_to_csv:
        write_data_to_pkl(X_train, Y_train, X_test, Y_test)
    return (X_train, Y_train), (X_test, Y_test)

def write_data_to_pkl(X_train, Y_train, X_test, Y_test, model_file="model.pkl"):
    data = {}
    data['x train'] = X_train
    data['y train'] = Y_train
    data['x test'] = X_test
    data['y test'] = Y_test
    data['company listings'] = HEADERS
    with open(model_file, "wb") as f:
        pickle.dump(data, f)


def load_data(model_file="model.pkl"):
    data = None
    with open(model_file, "rb") as f:
        data = pickle.load(f)
    return (data['x train'], data['y train']), (data['x test'], data['y test']), data['company listings']


def random_forest_classifier(X_train, Y_train, X_test):
    clf = RandomForestClassifier(max_depth=512, random_state=0)
    return clf.fit(X_train, Y_train).predict(X_test)


def correlate_daily_volume_price(n=10):
    coeffs = []
    for volume in volumes:
        coeffs.append(pearsonr(volume, price[0])[1])
    coeffs = np.array(coeffs)
    ind = np.argpartition(coeffs, -1*n)[-1*n:]
    ind = ind[np.argsort(coeffs[ind])]
    coeffs = np.array(coeffs)
    return ind


def plot_data(ylabel, Y_pred, Y_true, label):
    plt.ylabel(ylabel)
    plt.xlabel('time steps')
    plt.plot(Y_pred, label=label + ' prediction')
    plt.plot(Y_true, label=label + ' value')
    plt.legend()
    plt.tight_layout()
    plt.show()


def scrape_intra_day_data(data_range='1d', granularity='1m'):
    companies_unavailable = 0.0
    for h in HEADERS:
        jpy5m = get_quote_data(h, data_range, granularity)
        if jpy5m is not None:
            jpy5m = jpy5m.fillna(method='pad')
            INTRA_DAY_DATA.append(jpy5m)
        else:
            companies_unavailable = companies_unavailable + 1
            print("data for {} is unavailable".format(h))
    print("{} of companies are unavailable".format(100. * (companies_unavailable/len(HEADERS))))
    return INTRA_DAY_DATA


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = extract_from_csv()
    X_train, X_test = machine_learning_utils.z_score(X_train, X_test)
    headers = ['low', 'high', 'open']
    intra_day_data = scrape_intra_day_data()
    Y_pred, Y_train_pred = train_mlp_regressor(X_train, Y_train, X_test)
    lags = np.argmax(signal.correlate(Y_train_pred, Y_pred) - len(Y_pred))
    for i in range(len(headers)):
        plot_data(headers[i], Y_pred[:, i], Y_test[:, i], 'forecast S&P ' + headers[i])
        plot_data(headers[i], Y_train_pred[:, i], Y_train[:, i], 'S&P ' + headers[i])
