# Stock Price Prediction

## Description
The purpose of this project is to determine whether using historical stock indices can be helpful to predict the current stock price or stock price direction movement. <br>
The stock index of 'Apple Corporation' is considered for the analysis. <br>

## Data Set
Data is acquired on a daily basis for the last **7000** days and consists of opening, closing, high, low indices along with volume. The following diagram shows the indices accumulated over the last 7000 days. <br>

<div>     
         <img src="https://github.com/aa18514/machine_learning/blob/master/stock_price_prediction/images/opening_index.png", width="400" height="400" />     
         <img src="https://github.com/aa18514/machine_learning/blob/master/stock_price_prediction/images/closing_index.png", width="400" height="400" /> 
         <img src="https://github.com/aa18514/machine_learning/blob/master/stock_price_prediction/images/high_index.png", width="400" height="400" />
         <img src="https://github.com/aa18514/machine_learning/blob/master/stock_price_prediction/images/low_index.png", width="400" height="400" />
         <img src="https://github.com/aa18514/machine_learning/blob/master/stock_price_prediction/images/volume.png", width="400" height="400" />
         <img src="https://github.com/aa18514/machine_learning/blob/master/stock_price_prediction/images/market_capitalization.png", width="400" height="400" />
</div>

## Feature Engineering
Technical analysis was used to create features using stock indices. These include the following:
* [Williams % R](https://www.investopedia.com/terms/w/williamsr.asp)
* [Relative Strength Index](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/RSI)
* [7-day Rolling Standard Deviation](http://jonisalonen.com/2014/efficient-and-accurate-rolling-standard-deviation/)
* [Absolute Price Oscillator](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/apo)
* [Momentum](https://en.wikipedia.org/wiki/Momentum_(finance))
* [Commodity Channel Index](https://en.wikipedia.org/wiki/Commodity_channel_index)
* [Money Flow Index](http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:money_flow_index_mfi)
* [Average Directional Movement Index](https://en.wikipedia.org/wiki/Average_directional_movement_index)
* [2-day, 4-day and 16-day Simple Moving Average](https://www.investopedia.com/terms/s/sma.asp)
* High - Low
* Close - Open

In addition to past historical data of Apple, historical data of other companies listed in S&P 500 was also used. However, to prevent overfitting I only correlated the indices of Apple with the companies and used the 10 companies with the most positive correlation coefficients and the 10 companies with the most negative correlations. Furthermore, I also used the stock prices of the suppliers of Apple <br>

## Stratagies
* A feedforward neural network is used to train the model, where the target is raw stock indices. Although metrics such as MSE and R2 Score are low, these are incorrect indicators. This approach is flawed. Upon zooming in the predicted response it was discovered that the predicted response is just the delayed version of the true response.
* In order to fallacy indicated above, the raw prices were transformed to daily returns. <br>. A validation split of **0.33** was used. It was also discovered that the returns are normally distributed with mean **10.763** and standard deviation **3.735**, due to which z-score was used to standardize the data. <br>

After all of the features are engineered, the data is stored in pickle format, which allows other machine learning practictioners to train and test their models on feature engineered data-set <br>
