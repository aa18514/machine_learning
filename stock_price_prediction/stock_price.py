import csv
import os
import locale
import sys
import pickle
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from functools import reduce
from locale import atof
import sklearn.linear_model as d
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from scipy import signal
sys.path.insert(0, '..')
import machine_learning_utils
from pre_processor import pre_processor
from finance_transformer import optimizer
from matplotlib.axes import Axes
from data_acquisition import get_quote_data
from typing import Dict
import yaml
from nueral_network import nueral_network
from ransac import ransac
from sklearn.pipeline import Pipeline


DFS = list()
INTRA_DAY_DATA = list()
volumes = list()
price = list()
HEADERS = list()


def load_model(model_file='model\\regressor.yml'):
    cfg = None
    with open(model_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    return cfg


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
    Y_test = np.hstack([Y_test]).flatten()
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


def load_model(model_file='model\\regressor.yml'):
    cfg = None
    with open(model_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    return cfg


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


def scrape_yahoo_intra_day_data(cfg, data_range='3000d', granularity='1d', write_data_to_pkl=True, filename='intra_day_data'):
    companies_unavailable = 0.0
    data = get_quote_data('^GSPC', data_range, granularity)
    idx = data.index.unique()
    idx = pd.to_datetime(idx)
    INTRA_DAY_DATA = []
    print(HEADERS)
    company_quote = get_quote_data(HEADERS[1], data_range, granularity)
    company_quote = company_quote.reindex(idx, method='pad')
    company_quote = company_quote.fillna(method='pad')
    INTRA_DAY_DATA.append(company_quote)
    for company in cfg['suppliers']:
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
    return INTRA_DAY_DATA, data


def load_intra_day_data(filename='intra_day_data'):
    data_set = None
    with open(filename + '.pkl', "rb") as f:
        data_set = pickle.load(f)
    return data_set['hourly companies data'], data_set['s&p 500 data']


def add_suppliers(intra_day_data, us_suppliers=['JBL', 'MU', 'QCOM', 'DIOD', 'STM', 'TXN', 'ADI', 'GLUU'], range_days='3000d', interval='1d'):
    suppliers = []
    data = get_quote_data('^GSPC', range_days, interval)
    idx = data.index.unique()
    idx = pd.to_datetime(idx)
    for supplier in us_suppliers:
        company_quote = get_quote_data(supplier, range_days, interval)
        company_quote = company_quote.reindex(idx, method='pad')
        company_quote = company_quote.fillna(method='pad')
        suppliers.append(company_quote)
    return suppliers

 
def pre_process_hourly_data(cfg: Dict, 
                            intra_day_data,
                            apple_stock,
                            threshold=1,
                            predict_direction=False)->(Dict):
    look_ahead_days = int(cfg['look_ahead_days'])
    apple_stock = apple_stock.fillna(0)
    sp_data = apple_stock.values[look_ahead_days:]/apple_stock.values[:-look_ahead_days]
    if cfg['transformation'] == 'log':
        sp_data = np.log(sp_data) #log with base e
    else:
        pass
    if predict_direction:
        sp_data[sp_data <= threshold] = 0
        sp_data[sp_data > threshold] = 1
    finance_optimizer = optimizer(intra_day_data)
    transformation_dict = {
        'bollinger_bands' : finance_optimizer.compute_bb(i)
        'money_flow_index' : finance_optimizer.money_flow_index(i)
        'average_directional_index' : finance_optimizer.average_directional_movement_index(i)
        'momentum' : finance_optimizer.momentum(i)
        'hodrick_prescott' : finance_optimizer.calculate_hodrick_prescott(i),
        'trix' : finance_optimizer.trix(i)
        'relative_strength_index' : finance_optimizer.compute_rsi(i),
        'absolute_price_oscillator' : finance_optimizer.compute_absolute_price_oscillator(i)
    }
    for i in range(len(intra_day_data)):
        finance_optimizer.compute_high_low(i)
        finance_optimizer.compute_open_close(i)
        for keys in cfg['features']:
            if isinstance(keys, str):
                transformation_dict[keys]
            elif type(keys) is dict:
                for key, value in keys.items():
                    if key == 'rolling_standard_deviation':
                        for _, window_size in value.items():
                            finance_optimizer.compute_rolling_std(i, int(window_size))
                    elif key == 'moving_average':
                        for _, window_size in value.items():
                            finance_optimizer.calculate_moving_average(i, int(window_size))
        finance_optimizer.drop_data(i, ['close', 'low', 'volume'])
    stocks = finance_optimizer.merge()
    sp_data = sp_data[3:]
    stocks = stocks[3:-look_ahead_days]
    stocks = stocks.fillna(0)
    data_set = np.hstack([stocks.values, sp_data.reshape(len(stocks.values), 1)])
    data_set = data_set[:, 1:]
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


def feature_engineering(cfg: Dict, appl_stock, intra_day_data, HEADERS)->Dict:
    x_labels = ['days', 'days', 'days']
    y_labels =  ['closing minus opening index', 'high minus low index', 'volume', 'market cap']
    title = ['stock_index for AAPL'] * len(y_labels)
    y_values = [appl_stock['close'] - appl_stock['open'].values, appl_stock['high'] - appl_stock['low'].values, appl_stock['volume'], appl_stock['volume'].values * appl_stock['close'].values]
    for i in range(len(x_labels)):
        plt.xlabel(x_labels[i])
        plt.ylabel(y_labels[i])
        plt.title(title[i])
        plt.plot(y_values[i])
        plt.tight_layout()
        plt.show()
    #coerr = calculate_company_betas(HEADERS, intra_day_data[:-8], appl_stock[1:])
    #pos_companies, c_top, h_top = get_top_n_correlated_companies(coerr, HEADERS, intra_day_data[:-8])
    #neg_companies, c_bottom, h_bottom = get_bottom_n_correlated_companies(coerr, HEADERS, intra_day_data[:-8])
    #print(h_top)

    #print(h_bottom)
    companies = []
    #for company in pos_companies:
    #    companies.append(company)
    #for company in neg_companies:
    #    companies.append(company)
    #companies.append(intra_day_data[0])
    #for i in range(1, 20, 1):
    #    companies.append(intra_day_data[-1 * i])
    #print(len(companies))
    companies = intra_day_data
    data_set = pre_process_hourly_data(cfg, companies, appl_stock['close'])
    print(data_set)
    return data_set


def load_data(cfg: Dict)->Dict:
    intra_day_data = None
    data = None
    if cfg['scrape_yahoo_finance'] == True:
        intra_day_data, data = scrape_yahoo_intra_day_data(cfg)
    else:
        intra_day_data, data = load_intra_day_data()
    return intra_day_data, data


if __name__ == "__main__":
    cfg = load_model()['nn']
    train_test_split = float(cfg['train_test_split'])
    data_acquisition()
    HEADERS = np.array(HEADERS)
    intra_day_data, data = load_data(cfg)
   
    print("this is len of intra day data: {}".format(len(intra_day_data)))
    appl_stock = intra_day_data[0].fillna(0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    machine_learning_utils.histogram_residuals((appl_stock['open'].values[1:]/appl_stock['open'].values[:-1]), ax1)
    plt.show()
    data_set = feature_engineering(cfg, appl_stock, intra_day_data, HEADERS)
    print(train_test_split)
    (X_train, Y_train), (X_test, Y_test) = create_train_test_patches(data_set, 1, train_test_split)
    n = int(np.floor(train_test_split * (data_set.shape[0])))
    test_data = appl_stock['close'].values[n:]
    test_data = test_data[3:]
    pipeline = Pipeline([
        ('pre_processor', pre_processor('z_score')),
        ('model', nueral_network()),
        ])
    Y_pred = pipeline.fit(X_train, Y_train).predict(X_test)
    Y_pred = (Y_pred.flatten())
    Y_train_pred = (Y_train_pred.flatten())
    #visualize_classification(Y_pred, Y_train, Y_train_pred, Y_test)
    #print(np.sqrt(np.mean((Y_train_pred - Y_train)**2)))
    #print(np.sqrt(np.mean((Y_pred - Y_test)**2)))
    #print(np.mean(np.abs(100*(Y_pred - Y_test)/Y_test)))
    #plt.plot(Y_train_pred, label='predicted AAPL values in past')
    #plt.plot(Y_train, label='true AAPL values in past')
    Y_train_pred = np.exp(Y_train_pred)
    Y_train = np.exp(Y_train)
    Y_pred = np.exp(Y_pred)
    Y_test = np.exp(Y_test)
    Y_pred = np.append(Y_train_pred[-1], Y_pred)
    Y_test = np.append(Y_train[-1], Y_test)
    titles = ['Train Data', 'Test Data']
    xlabels = ['hours elapsed', 'hours elapsed']
    ylabel = ['stock_price_index', 'stock price index']
    action = [[Y_train_pred, Y_train], [Y_pred, Y_test]]
    label = [['predicted share price for AAPL', 'true share price for AAPL'],
             ['predicted share price for AAPL', 'true share price for AAPL']]
    
    start_point = 0
    end_point = 0
    for i in range(len(xlabels)):
        plt.xlabel(xlabels[i])
        plt.ylabel(ylabel[i])
        plt.title(titles[i])
        end_point = end_point + len(action[i][0])
        for j in range(len(action[i])):
            plt.plot(np.arange(start_point, end_point, 1), action[i][j], label=label[i][j])
        plt.legend()
        plt.tight_layout()
        plt.show()
        start_point = start_point + len(action[i][0])
    test_data = appl_stock['low'].values[n:]
    plt.xlabel('hours elapsed')
    plt.ylabel('stock price index')
    plt.plot(test_data[3:] * Y_pred, label='predicted share indices for AAPL')
    plt.plot(test_data[3:] * Y_test, label='true share indices for AAPL')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    neg_res = 0
    total_neg_res = 0
    pos_res = 0
    total_pos_res = 0
    Y_pred[Y_pred < 1] = 0
    Y_pred[Y_pred > 1] = 1
    Y_test[Y_test < 1] = 0
    Y_test[Y_test > 1] = 1
    for i in range(len(Y_test)): 
        if Y_test[i] == 1:
            total_pos_res = total_pos_res + 1
            if Y_pred[i] == 1:
                pass
            elif Y_pred[i] == 0:
                pos_res = pos_res + 1
        elif Y_test[i] == 0:
            total_neg_res = total_neg_res + 1
            if Y_pred[i] == 0:
                pass
            elif Y_pred[i] == 1:
                neg_res = neg_res + 1
    print(Y_pred != Y_test)
    print(np.sum(Y_pred != Y_test))
    print("this is life")
    print(np.sum(Y_pred != Y_test)/len(Y_test))
    print("pos: {}".format(pos_res/total_pos_res))
    print("neg: {}".format(neg_res/total_neg_res))
    """
    #plt.plot(Y_pred, Y_test, 'r+')
    #plt.show()
    #plt.xcorr(Y_pred, Y_test, normed=True, usevlines=True, maxlags = 30)
    #plt.show()
    lags = np.argmax(signal.correlate(Y_pred, Y_test)) - len(Y_pred)
    print(lags)
    print(r2_score(Y_pred, Y_test))
    print(r2_score(Y_test, Y_pred))
    corr = signal.correlate(Y_pred, Y_test)
    corr = corr/np.linalg.norm(corr)
    plt.plot(corr)
    plt.show()
    """
