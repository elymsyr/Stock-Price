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

def load_model_scaler():
    model = load_model('lstm_model_new.h5', compile=False)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    scaler = joblib.load('scaler_new.gz')
    return model, scaler

def get_scaled_data(df, scaler: MinMaxScaler, target_column: str = 'Close'):
    target_column_index = df.columns.tolist().index(target_column)
    scaled_data = scaler.fit_transform(df)
    return scaled_data, target_column_index

def create_dataset(data: np.ndarray, time_step: int=10):
    X = []
    for i in range(len(data) - time_step):
        # Define the range of input sequences
        end_ix = i + time_step
        
        # Ensure that the dataset is within bounds
        if end_ix > len(data)-1:
            break
            
        # Extract input and output parts of the pattern
        seq_x = data[i:end_ix]
        
        # Append the parts
        X.append(seq_x)
    return np.array(X), data.shape[1], time_step

def update_data_to_inverse(predicted_data: np.ndarray, scaler: MinMaxScaler, target_column_index: int, feature_number: int):
    new_dataset = np.zeros(shape=(len(predicted_data), feature_number))
    new_dataset[:,target_column_index] = predicted_data.flatten()
    return scaler.inverse_transform(new_dataset)[:, target_column_index].reshape(-1, 1)

def prediction():
    model, scaler = load_model_scaler()
    data_dict = get_data(day_before=100, order='a')
    df= format_data(data_dict)
    df = df.iloc[-20:]
    scaled_data, target_column_index = get_scaled_data(df=df, scaler=scaler)
    X, feature_number, time_step = create_dataset(data=scaled_data)
    predicted_data = model.predict(X)
    return predicted_data, df, model, scaler, target_column_index, feature_number