import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.src.callbacks import History
from keras.src.layers import LSTM, Dense

def get_df(data_size:int=100, path:str="Data\AAPL_stock_prices.csv", delimeter: str = ',', from_end: bool = True, date_column: str = 'Date', target_column: str = 'Close') -> tuple[np.ndarray, MinMaxScaler, pd.DataFrame]:
    df = pd.read_csv(path, delimiter=',')
    df = df.iloc[-data_size:, :] if from_end else df.iloc[:data_size, :]
    dates = pd.to_datetime(df[date_column])
    df.drop(columns=[date_column], inplace=True)
    df.index = dates

    scaler = MinMaxScaler(feature_range=(0, 1))
    merged_data = np.hstack((df[target_column].values.reshape(-1, 1), df.drop(columns=[target_column]).values))
    return scaler.fit_transform(merged_data), scaler, df

def create_dataset(data: np.ndarray, time_step: int=10, output_window_size: int = 1) -> tuple[np.ndarray, np.ndarray]:
    X, Y = [], []
    #(len, window_size, features)
    for i in range(len(data) - time_step):
        # Define the range of input sequences
        input_end_index = i + time_step
        
        # Define the range of output sequences
        output_end_index = input_end_index + output_window_size
        
        # Ensure that the dataset is within bounds
        if output_end_index > len(data)-1:
            break
            
        # Extract input and output parts of the pattern
        seq_x, seq_y = data[i:input_end_index], data[output_end_index]
        
        # Append the parts
        X.append(seq_x)
        Y.append(seq_y)

    return np.array(X), np.array(Y)  

def split_data(X: np.ndarray, Y: np.ndarray, train_size: float = 0.7, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = train_size, random_state = random_state)
    shapes: str = f"{X_train.shape=}\n{X_test.shape=}\n{Y_train.shape=}\n{Y_test.shape=}"
    print(shapes)
    return X_train, X_test, Y_train, Y_test

def create_model(input_shape: tuple, layers_with_units: list[int] = [128,128], optimizer: str = 'adam', loss: str = 'mean_squared_error') -> Sequential:
    # Create the LSTM model
    model = Sequential()
    for layer in layers_with_units:
        model.add(LSTM(layer, return_sequences=True, input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer = optimizer, loss = loss)
    return model

def show_loss(history: History):
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Value Loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
def predict(X_train, X_test, model: Sequential) -> tuple[np.ndarray, np.ndarray]:
    return model.predict(X_train), model.predict(X_test)

# Inverse transform the predictions
def update_data_to_inverse(predicted_data: np.ndarray, scaler: MinMaxScaler, target_column_index: int = 0):
    new_dataset = np.zeros(shape=(len(predicted_data), predicted_data.shape[2]))
    new_dataset[:,target_column_index] = predicted_data.flatten()
    return scaler.inverse_transform(new_dataset)[:, target_column_index].reshape(-1, 1)


data, scaler, *_ = get_df(data_size=500)
X: np.ndarray
Y: np.ndarray
X, Y = create_dataset(data=data)
X_train, X_test, Y_train, Y_test = split_data(X, Y)
print(f"{X_train.shape=}, {X_test.shape=}, {Y_train.shape=}, {Y_test.shape=}")
model = create_model(input_shape=(X_train.shape[1], X_train.shape[2]))
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1, batch_size=1, verbose=1)
# show_loss(history=history)

train_predict, test_predict = predict(X_train, X_test, model)

print(f"{train_predict.shape=}, {test_predict.shape=}, {Y_train.shape=}, {Y_test.shape=}")


train_predict = scaler.inverse_transform(update_data_to_inverse(train_predict, scaler=scaler))
test_predict = scaler.inverse_transform(update_data_to_inverse(test_predict, scaler=scaler))
Y_train = scaler.inverse_transform(update_data_to_inverse(Y_train, scaler=scaler))
Y_test = scaler.inverse_transform(update_data_to_inverse(Y_test, scaler=scaler))

print(f"{train_predict.shape=}, {test_predict.shape=}, {Y_train.shape=}, {Y_test.shape=}")