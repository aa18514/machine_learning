
import requests
import datetime
import pandas as pd
import arrow

def get_quote_data(symbol='iwm', data_range='100d', data_interval='1m', timezone='EST'):
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
            stock_quote = df.loc[:, ('open', 'high', 'low', 'close', 'volume')]
        catch Exception as e:
            print(e)
    else:
        print(data['chart']['error'])
    return stock_quote
