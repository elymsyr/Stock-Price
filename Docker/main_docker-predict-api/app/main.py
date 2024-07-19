import numpy as np
from json import loads
import matplotlib.pyplot as plt
import pandas as pd
from functions import *
from fastapi import FastAPI


app = FastAPI()

@app.get("/")
async def general():
    return {'status': 1}

@app.get("/info")
async def info():
    predicted_data, df, model, scaler, target_column_index, feature_number = prediction()
    return {"predicted_data": f"{predicted_data}", "df": f"{df}", "model": f"{model}", "scaler": f"{scaler}", "target_column_index": f"{target_column_index}", "feature_number": f"{feature_number}"}
    
@app.get("/save")
async def save(df, new_df):
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


@app.get("/predict")
async def main_run():
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
    df_json = new_df.to_json()
    return loads(df_json)

@app.get("/tomorrow")
async def tomorrow():
    predicted_data, df, model, scaler, target_column_index, feature_number = prediction()
    predicted_data = update_data_to_inverse(predicted_data=predicted_data, scaler=scaler, target_column_index=target_column_index, feature_number=feature_number)
    next_day = df.index.max() + pd.DateOffset(days=1)
    return {str(next_day).replace(' 00:00:00', ''): f"{(predicted_data[-1])}"}
