# %%
import numpy as np
import uvicorn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model # type: ignore
import joblib
from fastapi import FastAPI
from eodhd import APIClient
from datetime import datetime, timedelta
from datetime import datetime
API =  "6692ff520ca5b8.24552354" # eodhd 

# %%
def get_data(stock: str = 'AAPL', symbol: str = 'US', period: str = 'd', from_date = None, to_date = None, day_before: int = 20, order='a'):
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
    return data_dict

def format_data(data:dict):
    df = pd.DataFrame.from_dict(data, orient='index')
    df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

# %%
def load_model_scaler():
    model = load_model('lstm_model_test.keras', compile=False)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    scaler = joblib.load('scaler.gz')
    return model, scaler

# %%
def get_scaled_data(df, scaler: MinMaxScaler, target_column: str = 'Close'):
    target_column_index = df.columns.tolist().index(target_column)
    scaled_data = scaler.fit_transform(df)
    return scaled_data, target_column_index

# %%
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

# %%
# Inverse transform the predictions
def update_data_to_inverse(predicted_data: np.ndarray, scaler: MinMaxScaler, target_column_index: int, feature_number: int):
    new_dataset = np.zeros(shape=(len(predicted_data), feature_number))
    new_dataset[:,target_column_index] = predicted_data.flatten()
    return scaler.inverse_transform(new_dataset)[:, target_column_index].reshape(-1, 1)

app = FastAPI()
# %%
def prediction():
    model, scaler = load_model_scaler()
    data_dict = get_data(day_before=100, order='a')
    df= format_data(data_dict)
    df = df.iloc[-20:]
    scaled_data, target_column_index = get_scaled_data(df=df, scaler=scaler)
    X, feature_number, time_step = create_dataset(data=scaled_data)
    predicted_data = model.predict(X)
    return predicted_data, df, model, scaler, target_column_index, feature_number

app.get("/")
def general():
    return {'statur': 1}

app.get("/infor")
def info():
    predicted_data, df, model, scaler, target_column_index, feature_number = prediction()
    return {"predicted_data": predicted_data, "df": df, "model": model, "scaler": scaler, "target_column_index": target_column_index, "feature_number": feature_number}
    
app.get("/save")
def save(df, new_df):
    df.to_csv(f'predictions.csv')
    plt.plot(new_df['Close'], label='Close')
    plt.plot(new_df['Predicted_close'], label='Predicted Close')
    plt.plot(new_df['Desired_prediction'], label='Desired Prediction')
    plt.title('Predictons')
    plt.ylabel('Prices')
    plt.xlabel('Date')
    plt.legend()
    plt.savefig('figure.png')
    return {'check_save': 1}


app.get("/predict")
def main_run():
    predicted_data, df, model, scaler, target_column_index, feature_number = prediction()
    predicted_data = update_data_to_inverse(predicted_data=predicted_data, scaler=scaler, target_column_index=target_column_index, feature_number=feature_number)
    new_df = df[['Close']].iloc[-10:].copy()
    new_df['Predicted_close'] = predicted_data
    next_day = df.index.max() + pd.DateOffset(days=1)
    last_prediction = pd.DataFrame({'Close': [np.nan], 'Predicted_close': predicted_data[-1]}, index=[f"{next_day} 00:00:00"])
    desired_prediction = np.full((11,1), np.nan)
    new_df = pd.concat([new_df, last_prediction])
    desired_prediction[-2] = new_df['Close'][-2]
    desired_prediction[-1] = new_df['Predicted_close'][-1]
    desired_prediction = desired_prediction.reshape(-1,1)
    new_df['Desired_prediction'] = desired_prediction
    new_df.index = pd.to_datetime(new_df.index)
    new_df.index = new_df.index.strftime('%m-%d')
    return new_df.to_json()

app.get("/tomorrow")
def tomorrow():
    predicted_data, *_ = prediction()
    try: 
        return {0: predicted_data[-1]}
    except:
        return {0: 0}

# %%
if __name__ == "__main__":   
    uvicorn.run(app, host="127.0.0.1", port=8000)