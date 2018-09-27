from functools import reduce
import numpy as np
from pandas import Series
import pandas as pd

class optimizer:
    """a class that can calculate the technical indicators given the stock indices
    these technical indicators include williams_r, rolling standard deviation, absolute
    price oscillators, relative strength index"""

    def __init__(self, stock_data, keys=['williams_r', 'rsi']):
        """@stock_data: list of dataframes for S&P 500 Companies - this could be either intra-day, 
        daily, weekly or monthly data"""
        self._stock_data = stock_data
        for i in range(len(self._stock_data)):
            for key in keys:
                self._stock_data[i][key] = \
                        Series(np.random.randn(len(self._stock_data[i]['open'].values)), index=self._stock_data[i].index)

    
    def compute_williams_r(self, company_index, pad_value=0):
        c = 0
        company = self._stock_data[company_index]
        for index, row in self._stock_data[company_index].iterrows():
            if c > 14:
                self._stock_data[company_index].set_value(index, 'williams_r', ((max(company['high'][c - 14: c]) - row['close']) / (max(company['high'][c - 14: c]) - min(company['low'][c - 14 : c]))))
            else: 
                self._stock_data[company_index].set_value(index, 'williams_r', pad_value)
            c = c + 1

    
    def compute_rsi(self, company_index, span=14):
        delta = self._stock_data[company_index]['close'].diff()[1:]
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        roll_up = pd.DataFrame.ewm(up, span=span)
        roll_down = pd.DataFrame.ewm(down.abs(), span=span)
        relative_strength_index = roll_up.mean()/roll_down.mean()
        relative_strength_index = 100 - (100/(1.0 + relative_strength_index))
        self._stock_data[company_index]['rsi'][1:] = relative_strength_index


    def compute_rolling_std(self, company_index, span=7):
       self._stock_data[company_index]['std_dev'] = self._stock_data[company_index]['close'].rolling(7).std()


    def calculate_moving_average(self, company_index, period=2):
        moving_average = []
        for j in range(period-1):
            moving_average.append(self._stock_data[company_index]['close'].values[j])
        for j in range(period-1, len(self._stock_data[company_index]['close'].values) - 1):
            moving_average.append(np.mean(self._stock_data[company_index]['close'].values[j - (period - 1) : j + 1]))
        val = 0
        for j in range(1, period+1):
            val = val + self._stock_data[company_index]['close'].values[-1 * j]
        moving_average.append(float(val)/period)
        self._stock_data[company_index][str(period) + '-val'] = moving_average


    def compute_absolute_price_oscillator(self, company_index, slow='16-val', fast='2-val'):
        company = self._stock_data[company_index]
        self._stock_data[company_index]['delta'] = np.array(company[fast].values) - np.array(company[slow].values)

    
    def drop_data(self, company_index, keys=['close', 'low', 'volume']):
        for key in keys:
            self._stock_data[company_index] = self._stock_data[company_index].drop(key, axis=1)

    
    def compute_high_low(self, company_index): 
        self._stock_data[company_index]['high'] = self._stock_data[company_index]['high'].values - self._stock_data[company_index]['low'].values

    
    def compute_open_close(self, company_index):
        self._stock_data[company_index]['open'] = self._stock_data[company_index]['close'].values - self._stock_data[company_index]['close'].values

    def merge(self):
        stocks = reduce(lambda left, right: pd.merge(left, right, on='dt'), self._stock_data)
        stocks = stocks.fillna(method='bfill')
        return stocks
