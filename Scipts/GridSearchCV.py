import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras import Sequential
from keras.src.layers import LSTM, Dense, Input

data_size = 1000
df = pd.read_csv('Data\AAPL_stock_prices.csv', delimiter=',')
df = df.iloc[-data_size:, :]
dates = pd.to_datetime(df['Date'])
df.drop(columns=['Date'], inplace=True)
df.index = dates
target_column_index = df.columns.tolist().index('Close')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

def create_dataset(data: np.ndarray, time_step: int=10):
    X, Y = [], []
    for i in range(len(data) - time_step):
        # Define the range of input sequences
        end_ix = i + time_step
        
        # Define the range of output sequences
        out_end_ix = end_ix + 1
        
        # Ensure that the dataset is within bounds
        if out_end_ix > len(data)-1:
            break
            
        # Extract input and output parts of the pattern
        seq_x, seq_y = data[i:end_ix], data[out_end_ix]
        
        # Append the parts
        X.append(seq_x)
        Y.append(seq_y)
    return np.array(X), np.array(Y), data.shape[1], time_step


X, Y, feature_number, time_step = create_dataset(data=scaled_data)

print(X.shape, Y.shape)

# Split the data into training and testing sets
train_size = int(len(X) * 0.7)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

def create_lstm_model(neurons=64, loss='mean_squared_error', activation='linear', optimizer='Adam', batch_size=1, epochs=50):
    model = Sequential()
    model.add(LSTM(units=neurons, return_sequences=True, activation=activation, input_shape=(X.shape[1], feature_number)))
    model.add(LSTM(units=neurons, return_sequences=False))
    model.add(Dense(1))
    model.compile(loss=loss, optimizer=optimizer)
    
    return model
# param_grid = {
#     'model__optimizer': ['SGD','Adam'], #  'RMSprop', 
#     'model__loss': ['mean_squared_error', 'mean_absolute_error'],
#     'batch_size': [6, 12, 18, 24],
#     'epochs': [20, 50, 100],
#     'model__neurons': [16, 64, 128],
#     'model__second_layer':[6, 128, 256],  # 0 means no second LSTM layer
#     'model__activation': ['relu', 'tanh', 'sigmoid', 'linear']
# }

param_grid = {
    'model__optimizer': ['adam'], #  'RMSprop', 
    'model__loss': ['mean_squared_error'],
    'batch_size': [6],
    'epochs': [1],
    'model__neurons': [64, 128],
    'model__activation': ['linear'] # 'relu', 'tanh', 'sigmoid', 
}

param_grid = {
    'model__optimizer': ['adam'], #  'RMSprop', 
    'model__loss': ['mean_squared_error'],
    'batch_size': [6],
    'epochs': [1],
    'model__neurons': [64, 128],
    'model__activation': ['linear'] # 'relu', 'tanh', 'sigmoid', 
}



lstm_regressor = KerasRegressor(model=create_lstm_model, batch_size=1, epochs=1, verbose=1)
# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=lstm_regressor, param_grid=param_grid, scoring='neg_mean_absolute_error', error_score='raise', cv=3)
# Fit the grid search to the data
grid_search.fit(X_train, Y_train[:, target_column_index].reshape(-1,1))

print(grid_search.best_params_)

print(grid_search.best_score_)