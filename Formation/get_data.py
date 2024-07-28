import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model # type: ignore
import joblib
from eodhd import APIClient
from datetime import datetime, timedelta
from datetime import datetime
API =  "6692ff520ca5b8.24552354" # eodhd 

def get_data(stock: str = 'AAPL', symbol: str = 'US', period: str = 'd', from_date = None, to_date = None, day_before: int = 20, order='a'):
    current_date = datetime.now()
    formatted_current_date = current_date.strftime('%Y-%m-%d')
    date_10_days_before = current_date - timedelta(days=day_before)
    formatted_date_10_days_before = date_10_days_before.strftime('%Y-%m-%d')
    if from_date is None: from_date = formatted_date_10_days_before
    if to_date is None: to_date = formatted_current_date
    
    api = APIClient(API)

    resp = api.get_eod_historical_stock_market_data(symbol = f'{stock}.{symbol}', period=period, from_date = from_date, to_date = to_date, order=order)

    data_dict = {}

    for item in resp:
        data_dict[item.pop('date')] = item

    return data_dict

def format_data(data:dict):
    df = pd.DataFrame.from_dict(data, orient='index')
    df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def get_scaled_data(df, scaler: MinMaxScaler, target_column: str = 'Close'):
    target_column_index = df.columns.tolist().index(target_column)
    scaled_data = scaler.fit_transform(df)
    return scaled_data, target_column_index

def update_data_to_inverse(predicted_data: np.ndarray, scaler: MinMaxScaler, target_column_index: int, feature_number: int):
    new_dataset = np.zeros(shape=(len(predicted_data), feature_number))
    new_dataset[:,target_column_index] = predicted_data.flatten()
    return scaler.inverse_transform(new_dataset)[:, target_column_index].reshape(-1, 1)

def find_data_to_formation():
    data = format_data(get_data())
    return data[['Close']].to_numpy()