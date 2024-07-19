from eodhd import APIClient
from APIKEY import API
from datetime import datetime, timedelta
import pandas as pd


def get_data(stock: str = 'AAPL', symbol: str = 'US', period: str = 'd', from_date = None, to_date = None, day_before: int = 50, order='d'):
    current_date = datetime.now()
    formatted_current_date = current_date.strftime('%Y-%m-%d')
    date_10_days_before = current_date - timedelta(days=day_before)
    formatted_date_10_days_before = date_10_days_before.strftime('%Y-%m-%d')
    if from_date is None: from_date = formatted_date_10_days_before
    if to_date is None: to_date = formatted_current_date
    
    print(from_date, to_date)
    
    api = APIClient(API)

    resp = api.get_eod_historical_stock_market_data(symbol = f'{stock}.{symbol}', period=period, from_date = from_date, to_date = to_date, order=order)

    data_dict = {}

    for item in resp:
        data_dict[item.pop('date')] = item

    print(data_dict)
    return data_dict, formatted_current_date

def format_data(data:dict):
    df = pd.DataFrame.from_dict(data, orient='index')
    df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df.index = pd.to_datetime(df.index)
    return df


# import requests, re
# import yfinance as yf
# from APIKEY import API_KEY


# def get_data(symbol: str = 'AAPL', function: str = 'TIME_SERIES_DAILY_ADJUSTED', api_key: str = API_KEY):
#     # replace the "demo" apikey below with your own key from https://www.alphavantage.co/support/#api-key
#     url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}'
#     response = requests.get(url)
#     return response.json() if response.status_code == 200 else None, response.status_code


# msft = yf.Ticker("MSFT")

# # get all stock info
# print(msft.info)

# # get historical market data
# hist = msft.history(period="1mo")

# # show meta information about the history (requires history() to be called first)
# print(msft.history_metadata)

# print(hist)
