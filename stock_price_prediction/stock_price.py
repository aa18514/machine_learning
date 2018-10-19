import csv
import os
import locale
import sys
import arrow
import pickle
import datetime
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from functools import reduce
from locale import atof
from keras.layers import Dense
from keras.optimizers import Adam
import sklearn.linear_model as d
from scipy.stats import pearsonr
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from sklearn.metrics import r2_score
from scipy import signal
sys.path.insert(0, '..')
import machine_learning_utils
from finance_transformer import optimizer
from matplotlib.axes import Axes
import train_model

DFS = list()
INTRA_DAY_DATA = list()
volumes = list()
price = list()
HEADERS = list()


def get_quote_data(symbol='iwm', data_range='100d', data_interval='1m', timezone='EST'):
    print(symbol)
    res = requests.get('https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={data_range}&interval={data_interval}'.format(**locals()))
    data = res.json()
    stock_quote = None
    if data['chart']['error'] is None:
        try:
            body = data['chart']['result'][0]
            dt = datetime.datetime
            dt = pd.Series(map(lambda x: arrow.get(x).to(timezone).datetime.replace(tzinfo=None), body['timestamp']), name='dt')
            df = pd.DataFrame(body['indicators']['quote'][0], index=dt)
            dg = pd.DataFrame(body['timestamp'])
            stock_quote =  df.loc[:, ('open', 'high', 'low', 'close', 'volume')]
        except Exception as e:
            print(e)
    else:
        print(data['chart']['error'])
    return stock_quote


def rmse_vec(pred_y, true_y):
    return K.mean((pred_y - true_y)**2)


def percent_to_float(x: str)->str:
    return float(x.strip('%'))


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
        #data_sets[:, (-1 * i)][1: ] = machine_learning_utils.log_transformation(data_sets[:, (-1 * i)]
    return data_sets


def create_train_test_patches(data_sets, no, alpha=0.90):
    n = int(np.floor(alpha * data_sets.shape[0]))
    X_train = data_sets[:n][:, :(-1*no)]
    Y_train = []
    Y_test = []
    for i in range(1, no+1):
        Y_train.append(data_sets[:n][:, int(-1*i)])
        Y_test.append(data_sets[n:][:, int(-1*i)])
    Y_train = np.hstack([Y_train])
    Y_train = data_sets[:n][:, -1]
    X_test = data_sets[n:][:, :(-1*no)]
    Y_test = np.hstack([Y_test])
    return (X_train, Y_train), (X_test, Y_test)


def data_acquisition():
    fileNames = os.listdir('stocks')
    headers = []
    dates = pd.read_csv('stocks\\AAPL_data.csv')
    idx = dates['date'].unique()
    idx = pd.to_datetime(idx, format="%Y-%m-%d")
    for value in fileNames:
        print(value)
        if value != 'S&P 500 Historical Data.csv':
            HEADERS.append(value[:value.find('_')])
            extract_data(value[:value.find('_')+1], idx)
    stock_data = reduce(lambda left, right: pd.merge(left, right, on='date'), DFS)
    sp_data = pd.read_csv('stocks\\S&P 500 Historical Data.csv')
    return stock_data[1: ], sp_data[1: ]


def extract_from_csv(write_to_csv=True):
    keys = ["Low", "High", "Open"]
    stock_data, sp_data = data_acquisition()
    data_sets = pre_process_data(stock_data, sp_data, keys)
    indices = list()
    for i in range(len(keys) - 1, 0):
        indices.append(sp_data[keys[i]].str.replace(",", "").astype(float).values)
    (X_train, Y_train), (X_test, Y_test) = create_train_test_patches(data_sets, 4)
    if write_to_csv:
        write_data_to_pkl(X_train, Y_train, X_test, Y_test)
    return (X_train, Y_train), (X_test, Y_test)


def write_data_to_pkl(X_train, Y_train, X_test, Y_test, model_file="data_file.pkl"):
    data = {
            'x train' : X_train,
            'y train' : Y_train,
            'x test' : X_test,
            'y test' : Y_test,
            'company listings' : HEADERS
            }
    with open(model_file, "wb") as f:
        pickle.dump(data, f)


def load_data(model_file="model.pkl"):
    data = None
    with open(model_file, "rb") as f:
        data = pickle.load(f)
    return (data['x train'], data['y train']), (data['x test'], data['y test']), data['company listings']


def correlate_daily_volume_price(n=10):
    coeffs = []
    for volume in volumes:
        coeffs.append(pearsonr(volume, price[0])[1])
    coeffs = np.array(coeffs)
    ind = np.argpartition(coeffs, -1*n)[-1*n:]
    ind = ind[np.argsort(coeffs[ind])]
    coeffs = np.array(coeffs)
    return ind


def plot_data(ylabel: str, Y_pred, Y_true, label):
    plt.ylabel(ylabel)
    plt.xlabel('time steps')
    plt.plot(Y_pred, label=label + ' prediction')
    plt.plot(Y_true, label=label + ' value')
    plt.legend()
    plt.tight_layout()
    plt.show()


def scrape_yahoo_intra_day_data(data_range='600d', granularity='1h', write_data_to_pkl=True, filename='intra_day_data'):
    companies_unavailable = 0.0
    data = get_quote_data('^GSPC', data_range, granularity)
    idx = data.index.unique()
    idx = pd.to_datetime(idx)
    INTRA_DAY_DATA = []
    for company in HEADERS:
        company_quote = get_quote_data(company, data_range, granularity)
        if company_quote is not None:
            company_quote = company_quote.reindex(idx, method='pad')
            company_quote = company_quote.fillna(method='pad')
            INTRA_DAY_DATA.append(company_quote)
        else:
            companies_unavailable = companies_unavailable + 1
            print("data for {} is unavailable".format(company))
    print("{} of companies are unavailable".format(100. * (companies_unavailable/len(HEADERS))))
    if write_data_to_pkl is True:
        data_set = {
                     'hourly companies data' : INTRA_DAY_DATA,
                     's&p 500 data' : data
                   }
        with open(filename + '.pkl', "wb") as f:
            pickle.dump(data_set, f)
    INTRA_DAY_DATA = add_suppliers(INTRA_DAY_DATA)        
    return INTRA_DAY_DATA, data


def load_intra_day_data(filename='intra_day_data'):
    data_set = None
    with open(filename + '.pkl', "rb") as f:
        data_set = pickle.load(f)
    return data_set['hourly companies data'], data_set['s&p 500 data']


def add_suppliers(intra_day_data, us_suppliers=['JBL', 'MU', 'QCOM', 'DIOD', 'STM', 'TXN', 'ADI', 'GLUU']):
    data = get_quote_data('^GSPC', '600d', '1h')
    idx = data.index.unique()
    idx = pd.to_datetime(idx)
    for supplier in us_suppliers:
        company_quote = get_quote_data(supplier, '600d', '1h')
        company_quote = company_quote.reindex(idx, method='pad')
        company_quote = company_quote.fillna(method='pad')
        intra_day_data.append(company_quote)
    return intra_day_data


def pre_process_hourly_data(intra_day_data, apple_stock, predict_direction=False):
    apple_stock = apple_stock.fillna(0)
    sp_data = np.log(apple_stock.values[1:]/apple_stock.values[:-1])
    if predict_direction: 
        sp_data[sp_data <= 0] = 0
        sp_data[sp_data > 0] = 1
    finance_optimizer = optimizer(intra_day_data)
    for i in range(len(intra_day_data)):
        finance_optimizer.calculate_money_flow_index(i)
        #finance_optimizer.average_directional_movement_index(i)
        #finance_optimizer.momentum(i)
        #finance_optimizer.compute_commodity_channel_index(i)
        #finance_optimizer.compute_williams_r(i)
        finance_optimizer.compute_high_low(i)
        finance_optimizer.compute_open_close(i)
        finance_optimizer.compute_rsi(i)
        finance_optimizer.compute_rolling_std(i)
        finance_optimizer.calculate_moving_average(i, 2)
        finance_optimizer.calculate_moving_average(i, 16)
        finance_optimizer.calculate_moving_average(i, 90)
        finance_optimizer.compute_absolute_price_oscillator(i)
        finance_optimizer.drop_data(i, ['close', 'low', 'volume'])
    stocks = finance_optimizer.merge()
    stocks = stocks[3:-1]
    print(sp_data)
    sp_data = sp_data[3:]
    print(sp_data)
    print(len(stocks.values))
    stocks = stocks.fillna(0)
    data_set = np.hstack([stocks.values, sp_data.reshape(len(stocks.values), 1)])
    return data_set


def calculate_company_betas(HEADERS, intra_day_data, aapl_stock, keys=['close', 'open', 'high', 'low']):
    coerr = []
    for i in range(len(HEADERS)):
        if HEADERS[i] == 'AAPL':
            pass
        elif HEADERS[i] != 'AAPL':
            stock = intra_day_data[i]
            stock = stock[:-1]
            stock = stock.fillna(0)
            beta = 0
            for key in keys:
                beta = beta + pearsonr(stock[key].values, aapl_stock[key].values)[0]
            beta = beta/len(keys)
            coerr.append(beta)
    coerr = np.array(coerr)
    return coerr


def get_top_n_correlated_companies(coerr, HEADERS, intra_day_data, n=10):
    ind = np.argpartition(coerr, -1*n)[-1*n: ]
    ind = ind[np.argsort(coerr[ind])]
    pos_companies = []
    for i in range(len(ind)):
        pos_companies.append(intra_day_data[ind[i]])
    return pos_companies, coerr[ind], HEADERS[ind]


def get_bottom_n_correlated_companies(coerr, HEADERS, intra_day_data, n=10):
    index = np.where(coerr < 0)
    headers_neg = HEADERS[index]
    coerr_1 = coerr[coerr < 0]
    intra_day_data_neg = []
    for i in range(len(index[0])):
        intra_day_data_neg.append(intra_day_data[index[0][i]])
    coerr_2 = np.abs(coerr_1)
    ind = np.argpartition(coerr_2, -1*n)[-1*n:]
    ind = ind[np.argsort(coerr_2[ind])]
    neg_companies = []
    for i in range(len(ind)):
        neg_companies.append(intra_day_data[ind[i]])
    return neg_companies, coerr_1[ind], headers_neg[ind]



def visualize_classification(Y_pred, Y_train, Y_train_pred, Y_test):
    print(Y_train_pred)
    print(Y_pred)
    print(Y_test)
    print(np.sum(Y_train_pred != Y_train)/len(Y_train))
    print(np.sum(Y_pred != Y_test)/len(Y_test))
    ab = (Y_pred != Y_test)
    values = []
    for i in ab[0]:
        if i == True:
            values.append(1)
        else:
            values.append(0)
    plt.plot(values, 'r*')
    plt.show()


def feature_engineering(appl_stock, intra_day_data, HEADERS):
    plt.xlabel('days')
    plt.ylabel('opening index')
    plt.title('stock index for AAPL')
    plt.plot(appl_stock['open'].values)
    plt.tight_layout()
    plt.show()
    plt.xlabel('days')
    plt.ylabel('closing index')
    plt.title('stock index for AAPL')
    plt.plot(appl_stock['close'].values)
    plt.tight_layout()
    plt.show()
    plt.xlabel('days')
    plt.ylabel('high index')
    plt.title('stock index for AAPL')
    plt.plot(appl_stock['high'].values)
    plt.tight_layout()
    plt.show()
    plt.xlabel('days')
    plt.ylabel('low index')
    plt.title('stock index for AAPL')
    plt.plot(appl_stock['low'].values)
    plt.tight_layout()
    plt.show()
    plt.xlabel('days')
    plt.ylabel('volume')
    plt.title('stock index for AAPL')
    plt.plot(appl_stock['volume'].values)
    plt.tight_layout()
    plt.show()
    plt.xlabel('days')
    plt.ylabel('market cap')
    plt.title('stock index for AAPL')
    plt.plot(appl_stock['volume'].values * appl_stock['close'].values)
    plt.tight_layout()
    plt.show()
    coerr = calculate_company_betas(HEADERS, intra_day_data, appl_stock[1:])
    pos_companies, c_top, h_top = get_top_n_correlated_companies(coerr, HEADERS, intra_day_data)
    neg_companies, c_bottom, h_bottom = get_bottom_n_correlated_companies(coerr, HEADERS, intra_day_data)
    print(h_top)
    print(h_bottom)
    companies = []
    for company in pos_companies:
        company = company.fillna(0)
        companies.append(company)
    for company in neg_companies:
        company = company.fillna(0)
        companies.append(company)
    companies.append(intra_day_data[1])
    data_set = pre_process_hourly_data(companies, appl_stock['low'])
    print(data_set)
    return data_set


if __name__ == "__main__":
    data_acquisition()
    HEADERS = np.array(HEADERS)
    intra_day_data, data = load_intra_day_data()
    appl_stock = intra_day_data[1].fillna(0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    machine_learning_utils.histogram_residuals((appl_stock['open'].values[1:]/appl_stock['open'].values[:-1]), ax1)
    plt.show()
    data_set = feature_engineering(appl_stock, intra_day_data, HEADERS)
    (X_train, Y_train), (X_test, Y_test) = create_train_test_patches(data_set, 1)
    print(X_train[0])
    #write_data_to_pkl(X_train, Y_train, X_test, Y_test)
    Y_test = Y_test.flatten()
    X_train, X_test = machine_learning_utils.z_score(X_train, X_test)
    Y_pred, Y_train_pred = train_model.train_mlp_regressor(X_train, Y_train, X_test)
    Y_pred = np.exp(Y_pred.flatten())
    Y_train_pred = np.exp(Y_train_pred.flatten())
    Y_train = np.exp(Y_train)
    Y_test = np.exp(Y_test)
    #visualize_classification(Y_pred, Y_train, Y_train_pred, Y_test)
    
    print(np.sqrt(np.mean((Y_train_pred - Y_train)**2)))
    print(np.sqrt(np.mean((Y_pred - Y_test)**2)))
    print(np.mean(np.abs(100*(Y_pred - Y_test)/Y_test)))
    #plt.plot(Y_train_pred, label='predicted AAPL values in past')
    #plt.plot(Y_train, label='true AAPL values in past')
    Y_pred = np.append(Y_train_pred[-1], Y_pred)
    Y_test = np.append(Y_train[-1], Y_test)
    plt.title('Train Data')
    plt.xlabel('hours elapsed')
    plt.ylabel('stock price index')
    plt.plot(Y_train_pred, label='predicted share price for AAPL')
    plt.plot(Y_train, label='true share price for AAPL')
    plt.legend()
    plt.show()
    plt.title('Test Data')
    plt.xlabel('hours elapsed')
    plt.ylabel('stock price index')
    plt.plot(np.arange(len(Y_train), len(Y_train) + len(Y_pred), 1), Y_pred, label='predicted share indices for AAPL')
    plt.plot(np.arange(len(Y_train), len(Y_train) + len(Y_pred), 1), Y_test, label='true share indices for AAPL')
    #plt.plot(np.arange(0, len(Y_test.flatten()) - 60), Y_test.flatten()[60:], label='S&P values shifted by 60 hours')
    plt.legend()
    plt.tight_layout()
    plt.show()
    n = int(np.floor(0.90 * (data_set.shape[0])))
    test_data = appl_stock['low'].values[n:]
    #test_data = test_data[:-1]
    plt.xlabel('hours elapsed')
    plt.ylabel('stock price index')
    plt.plot(test_data[3:] * Y_pred, label='predicted share indices for AAPL')
    plt.plot(test_data[3:] * Y_test, label='true share indices for AAPL')
    plt.legend()
    plt.tight_layout()
    plt.show()
    Y_pred[Y_pred < 1] = 0
    Y_pred[Y_pred > 1] = 1
    Y_test[Y_test < 1] = 0
    Y_test[Y_test > 1] = 1
    print(Y_pred)
    print(Y_test)
    print("this is life")
    print(Y_pred != Y_test)
    print(np.sum(Y_pred != Y_test))
    print(np.sum(Y_pred != Y_test)/len(Y_test))
  
    #plt.plot(Y_pred, Y_test, 'r+')
    #plt.show()
    #plt.xcorr(Y_pred, Y_test, normed=True, usevlines=True, maxlags = 30)
    #plt.show()
    lags = np.argmax(signal.correlate(Y_pred, Y_test)) - len(Y_pred)
    print(lags)
    print(r2_score(Y_pred, Y_test))
    print(r2_score(Y_test, Y_pred))
    corr = signal.correlate(Y_pred, Y_test)
    corr = corr/np.lingalg.norm(corr)
    plt.plot(corr)
    plt.show()
    #for i in range(len(headers)):
    #    plot_data(headers[i], Y_pred[:, i], Y_test[:, i], 'forecast S&P ' + headers[i])
    #    plot_data(headers[i], Y_train_pred[:, i], Y_train[:, i], 'S&P ' + headers[i])
    #"""
