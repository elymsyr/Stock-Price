# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import joblib
from eodhd import APIClient
from datetime import datetime, timedelta
API = "6692ff520ca5b8.24552354"


def get_data(stock: str = 'AAPL', symbol: str = 'US', period: str = 'd', from_date = None, to_date = None, day_before: int = 20):
    current_date = datetime.now()
    formatted_current_date = current_date.strftime('%Y-%m-%d')
    date_10_days_before = current_date - timedelta(days=day_before)
    formatted_date_10_days_before = date_10_days_before.strftime('%Y-%m-%d')
    if from_date is None: from_date = formatted_date_10_days_before
    if to_date is None: to_date = formatted_current_date
    
    print(from_date, to_date)
    
    api = APIClient(API)

    resp = api.get_eod_historical_stock_market_data(symbol = f'{stock}.{symbol}', period=period, from_date = from_date, to_date = to_date, order='a')

    data_dict = {}

    for item in resp:
        data_dict[item.pop('date')] = item

    print(data_dict)
    return data_dict

def format_data(data:dict):
    df = pd.DataFrame.from_dict(data, orient='index')
    df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

# %%
model = load_model('lstm_model_test.keras')
scaler = joblib.load('scaler.gz')

# %%
df : pd.DataFrame = format_data(data = get_data())

# %%
print(df)

# %%
def get_scaled_data(df: pd.DataFrame, scaler: MinMaxScaler, data_size:int=500, path:str=r"..\Data\AAPL_stock_prices.csv", delimeter: str = ',', from_end: bool = True, date_column: str = 'Date', target_column: str = 'Close'):
    target_column_index = df.columns.tolist().index(target_column)
    scaled_data = scaler.fit_transform(df)
    return scaled_data, target_column_index

# %%
scaled_data, target_column_index = get_scaled_data(df=df, scaler=scaler)

# %%
def create_dataset(data: np.ndarray, time_step: int=10):
    X = []
    for i in range(len(data) - time_step):
        # Define the range of input sequences
        end_ix = i + time_step
        
        # Define the range of output sequences
        out_end_ix = end_ix
        
        # Ensure that the dataset is within bounds
        if out_end_ix > len(data)-1:
            break
            
        # Extract input and output parts of the pattern
        seq_x = data[i:end_ix]
        
        # Append the parts
        X.append(seq_x)
    return np.array(X), data.shape[1], time_step

X, feature_number, time_step = create_dataset(data=scaled_data)
print(df)
print(X[:2])

# %%
predicted_data = model.predict(X)
print(predicted_data)


