import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras import Sequential
from keras.src.layers import LSTM, Dense
from random import randint


settings: dict = {
    0: {
        'epoch' : 50,
        'batch' : 1,
        'layers' : [256,256],
        'optimizer' : 'adam',
        'loss' : 'mean_squared_error'        
    },
    1: {
        'epoch' : 50,
        'batch' : 1,
        'layers' : [128, 128],
        'optimizer' : 'adam',
        'loss' : 'mean_squared_error'        
    },
    2: {
        'epoch' : 100,
        'batch' : 2,
        'layers' : [256,256],
        'optimizer' : 'adam',
        'loss' : 'mean_squared_error'        
    },
    3: {
        'epoch' : 50,
        'batch' : 1,
        'layers' : [256,256,256],
        'optimizer' : 'adam',
        'loss' : 'mean_squared_error'        
    },
    4: {
        'epoch' : 50,
        'batch' : 1,
        'layers' : [256,256,128],
        'optimizer' : 'adam',
        'loss' : 'mean_squared_error'        
    },
    5: {
        'epoch' : 50,
        'batch' : 1,
        'layers' : [512],
        'optimizer' : 'adam',
        'loss' : 'mean_squared_error'        
    },
    6: {
        'epoch' : 50,
        'batch' : 1,
        'layers' : [512, 512],
        'optimizer' : 'adam',
        'loss' : 'mean_squared_error'        
    },
    7: {
        'epoch' : 50,
        'batch' : 1,
        'layers' : [256, 512],
        'optimizer' : 'adam',
        'loss' : 'mean_squared_error'        
    },    
}

def get_df(data_size:int=2000, path:str="Data\AAPL_stock_prices.csv", delimeter: str = ',', from_end: bool = True, date_column: str = 'Date', target_column: str = 'Close') -> tuple[np.ndarray, MinMaxScaler, int]:
    df = pd.read_csv(path, delimiter=delimeter)
    df = df.iloc[-data_size:, :] if from_end else df.iloc[:data_size, :]
    dates = pd.to_datetime(df[date_column])
    df.drop(columns=[date_column], inplace=True)
    df.index = dates

    target_column_index = df.columns.tolist().index(target_column)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler, target_column_index

scaled_data, scaler, target_column_index = get_df()

def create_model(input_shape: tuple, layers_with_units: list[int] = [128,128,64], optimizer: str = 'adam', loss: str = 'mean_squared_error') -> Sequential:
    # Create the LSTM model
    model = Sequential()
    for layer in layers_with_units[:-1]:
        model.add(LSTM(layer, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(layers_with_units[-1], return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer = optimizer, loss = loss)
    return model

def log(epoch, layers_with_units, optimizer, loss, train_mse = None, train_r2 = None, y_true = None, train_predict = None, test_mse = None, test_r2 = None, number = None):
    with open('log.txt', 'a') as file:
        if number or number == 0:
            file.write(f"  Order : {number}")
        file.write(f"\n  Train Results with Epoch - {epoch}:")
        if y_true and train_predict:
            file.write(f"\n  Y_True:\n{y_true}\nY_Predicted\n{train_predict}")        
        file.write(f"\n    Layers: {layers_with_units}")
        file.write(f"\n    Optimizer : {optimizer}")
        file.write(f"\n    Loss: {loss}")
        file.write(f"\n    Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
        file.write(f"\n    Train R2 Score: {train_r2:.4f}, Test R2 Score: {test_r2:.4f}\n\n")    
    
    

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

# Inverse transform the predictions
def update_data_to_inverse(predicted_data: np.ndarray, scaler: MinMaxScaler, target_column_index: int, feature_number: int):
    new_dataset = np.zeros(shape=(len(predicted_data), feature_number))
    new_dataset[:,target_column_index] = predicted_data.flatten()
    return scaler.inverse_transform(new_dataset)[:, target_column_index].reshape(-1, 1)


def start_search():
    train_number = randint(111111, 999999)
    scaled_data, scaler, target_column_index = get_df()
    X, Y, feature_number, time_step = create_dataset(data=scaled_data)
    train_size = int(len(X) * 0.7)
    test_size = len(X) - train_size
    
    results: dict = {}
    mse: dict = {}
    with open('log.txt', 'a') as file:
        file.write(f'Train {train_number}: \n')
    for order, setting in settings.items():
        X_train, X_test = X[0:train_size], X[train_size:len(X)]
        Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]        
        epoch = setting['epoch']
        batch = setting['batch']
        layers = setting['layers']
        optimizer = setting['optimizer']
        loss = setting['loss']
        print(f"\nTrain {order} with :\n   Epoch: {epoch}\n    Layers: {layers}\n    Batch: {batch}\n    Optimizer: {optimizer}\n    Loss: {loss}")
        model = create_model(input_shape=(X_train.shape[1], feature_number), layers_with_units=layers, optimizer=optimizer, loss=loss)
        
        history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epoch, batch_size=batch, verbose=1)

        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
           
        train_predict = update_data_to_inverse(predicted_data=train_predict, scaler=scaler, target_column_index=target_column_index, feature_number=feature_number)
        test_predict = update_data_to_inverse(predicted_data=test_predict, scaler=scaler, target_column_index=target_column_index, feature_number=feature_number)
        Y_train = scaler.inverse_transform(Y_train)
        Y_test = scaler.inverse_transform(Y_test)    
        
        train_mse = mean_squared_error(Y_train[:, target_column_index].reshape(-1, 1), train_predict)
        test_mse = mean_squared_error(Y_test[:, target_column_index].reshape(-1, 1), test_predict)
        train_r2 = r2_score(Y_train[:, target_column_index].reshape(-1, 1), train_predict)
        test_r2 = r2_score(Y_test[:, target_column_index].reshape(-1, 1), test_predict)        
    
        results[order] = [train_mse,test_mse,train_r2,test_r2]
        mse[order] = float(f"{float(train_mse):.5f}")
        log(number = order, epoch=epoch, layers_with_units=layers, optimizer=optimizer, loss=loss, train_mse=train_mse, train_r2=train_r2, test_mse = test_mse, test_r2 = test_r2)    
    resulted = {}
    sorted_dict = {k: v for k, v in sorted(mse.items(), key=lambda item: item[1])}
    for k, v in sorted_dict.items():
        resulted[k] = (mse[k], results[k])

    with open('results.txt', 'a') as file:
        file.write(f'Results for Train {train_number}: ')
        for key, value in resulted.items():
            file.write(f"\n  {key}_epoch:{settings[key]['epoch']}_layers:{settings[key]['layers']}_batch:{settings[key]['batch']}_optimizer:{settings[key]['optimizer']}_loss:{settings[key]['loss']}\n    {value}")
            print(f"\n  {key}_epoch:{settings[key]['epoch']}_layers:{settings[key]['layers']}_batch:{settings[key]['batch']}_optimizer:{settings[key]['optimizer']}_loss:{settings[key]['loss']}\n    {value}")
        file.write('\n\n')


start_search()