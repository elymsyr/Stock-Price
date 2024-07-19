import numpy as np
import pandas as pd
import argparse, requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras import Sequential
from sklearn.model_selection import train_test_split
from keras.src.layers import LSTM, Dense
from datetime import datetime
from random import randint
from os import scandir, makedirs
from os.path import join
from time import time
import joblib

parser = argparse.ArgumentParser(description='Process some parameters.')
parser.add_argument('--epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--random-state', type=int, default=42, help='Number of epochs')
parser.add_argument('--no-log', action='store_true', help='Disable logging and saving model.')
parser.add_argument('--url', type=str, default='https://raw.githubusercontent.com/elymsyr/Stock-Price/main/Data/AAPL_stock_prices.csv', help='Url to csv data file')
parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
parser.add_argument('--data-size', type=int, help='Data size')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer type')
parser.add_argument('--loss', type=str, default='mean_squared_error', help='Loss function')
parser.add_argument('--metrics', nargs='+', default=['mse'], help='List of metrics')
parser.add_argument('--layers', nargs='+', type=int, default=[256, 256], help='List of layers')
args = parser.parse_args()

RANDOM = args.random_state if args.random_state != 0 else randint(11,101)
LOG = not args.no_log
EPOCH = args.epoch
BATCH_SIZE = args.batch_size
LAYERS = args.layers
OPTIMIZER = args.optimizer
LOSS = args.loss
METRICS = args.metrics
DATA_SIZE = args.data_size
URL = args.url

def get_df_from_url() -> pd.DataFrame|None:
    if URL != None:
        response = requests.get(URL)
        if response.status_code == 200:
            # Save the downloaded CSV file
            with open('last_downloaded_data.csv', 'wb') as f:
                f.write(response.content)
            print(f"CSV file downloaded successfully from {URL}")

            # Read CSV file using pandas
            df = pd.read_csv('last_downloaded_data.csv')
            return df
        else:
            print(f"Failed to download CSV file from {URL}. Status code: {response.status_code}")
    return None

def get_df(delimeter: str = ',', from_end: bool = True, date_column: str = 'Date', target_column: str = 'Close') -> tuple[np.ndarray, MinMaxScaler, int, pd.DataFrame]:
    df: pd.DataFrame | None= get_df_from_url()
    if df is None:
        exit(code=99)
    if DATA_SIZE != None:
        df = df.iloc[-DATA_SIZE:, :] if from_end else df.iloc[:DATA_SIZE, :]
    dates = pd.to_datetime(df[date_column])
    df.drop(columns=[date_column], inplace=True)
    df.index = dates #type:ignore
    target_column_index = df.columns.tolist().index(target_column)
    try: 
        scaler = joblib.load('scaler.gz')
    except:
        scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler, target_column_index, df

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

def create_model(input_shape: tuple, layers_with_units: list[int] = [128,128,64], optimizer: str = 'adam', loss: str = 'mean_squared_error', metrics: list[str]=['mse', 'mape']) -> Sequential:
    # Create the LSTM model
    model = Sequential()
    for layer in layers_with_units[:-1]:
        model.add(LSTM(layer, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(layers_with_units[-1], return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer = optimizer, loss = loss, metrics=metrics)
    return model

def update_data_to_inverse(predicted_data: np.ndarray, scaler: MinMaxScaler, target_column_index: int, feature_number: int):
    new_dataset = np.zeros(shape=(len(predicted_data), feature_number))
    new_dataset[:,target_column_index] = predicted_data.flatten()
    return scaler.inverse_transform(new_dataset)[:, target_column_index].reshape(-1, 1)

def log_save(plot: list, number, time, model, scaler, epoch, layers_with_units, optimizer, loss_eval, loss, mae, test_mse, test_r2, df, train_mse = None, train_r2 = None):
    save_models(number, model, scaler)
    plot_results(plot[0][:, plot[1]], plot[2][:, plot[1]], plot[3], plot[4], number=number)
    new_folder_path = join('Models', str(number))
    current_datetime = datetime.now()
    current_datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    with open(f'{new_folder_path}\\log.txt', 'a') as file:
        file.write(f"{number} Train Results at {current_datetime_str}:")  
        file.write(f"\n    Elapsed Time: {time:.6f}")
        file.write(f"\n    Url: {URL}")
        file.write(f"\n    Epoch: {epoch}")
        file.write(f"\n    Random State: {RANDOM}")
        file.write(f"\n    Data Size: {len(df)}")
        file.write(f"\n    Layers: {layers_with_units}")
        file.write(f"\n    Optimizer: {optimizer}")
        file.write(f"\n    Loss: {loss}")
        file.write(f"\n    From Evaluations:\n        Loss: {loss_eval}\n        Mae: {mae}")
        file.write(f"\n    MSE:\n        Train MSE: {train_mse:.4f}\n        Test MSE: {test_mse:.4f}")
        file.write(f"\n    R2\n        Train R2 Score: {train_r2:.4f}\n        Test R2 Score: {test_r2:.4f}")

def save_models(number, model, scaler):
    new_folder_path = join('Models', str(number))
    try:
        makedirs(new_folder_path)
        joblib.dump(scaler, f'{new_folder_path}\\scaler.gz')
        model.save(f'{new_folder_path}\\model.keras')
        model.save(f'{new_folder_path}\\model.h5')
    except:
        joblib.dump(scaler, f'Models/unknown_number\\scaler.gz')
        model.save(f'Models/unknown_number\\model.keras')
        model.save(f'Models/unknown_number\\model.h5')
    
def check_number_availability(number: int):
    try:
        with scandir('Models') as entries:
            existing_names = [entry.name for entry in entries]
            while str(number) in existing_names:
                number = randint(10000000, 99999999)
            return number
    except FileNotFoundError:
        print("The directory does not exist.")
    except PermissionError:
        print("You do not have permissions to access this directory.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
def plot_results(Y_train, Y_test, train_predict, test_predict, number):
    plt.figure(figsize=(16, 10))

    # Plot training data and predictions
    plt.subplot(2, 1, 1)
    plt.plot(Y_train, 'b-', label='Training Data')
    plt.plot(train_predict, 'r-', label='Training Prediction')
    plt.title('Training Data vs Training Prediction')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()

    # Plot test data and predictions
    plt.subplot(2, 1, 2)
    plt.plot(Y_test, 'b-', label='Test Data')
    plt.plot(test_predict, 'r-', label='Test Prediction')
    plt.title('Test Data vs Test Prediction')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()

    plt.tight_layout()
    if LOG: plt.savefig(join('Models', str(number), 'figure.png'))

def run_model_train():
    start_time = time()
    scaled_data, scaler, target_column_index, df = get_df()
    X, Y, feature_number, time_step = create_dataset(data=scaled_data)
    # Split the data: 70% training, 15% validation, 15% testing
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=RANDOM)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=RANDOM)

    model = create_model(input_shape=(X_train.shape[1], feature_number), layers_with_units=LAYERS, optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS) # type: ignore
    # Train the model
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1) #type:ignore
    loss_eval, mae = model.evaluate(X_val, Y_val)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    # Predict
    train_predict = update_data_to_inverse(predicted_data=train_predict, scaler=scaler, target_column_index=target_column_index, feature_number=feature_number)
    test_predict = update_data_to_inverse(predicted_data=test_predict, scaler=scaler, target_column_index=target_column_index, feature_number=feature_number)
    Y_train = scaler.inverse_transform(Y_train)
    Y_test = scaler.inverse_transform(Y_test)

    # Calculate MSE
    train_mse = mean_squared_error(Y_train[:, target_column_index].reshape(-1, 1), train_predict)
    test_mse = mean_squared_error(Y_test[:, target_column_index].reshape(-1, 1), test_predict)

    # Calculate R2 score
    train_r2 = r2_score(Y_train[:, target_column_index].reshape(-1, 1), train_predict)
    test_r2 = r2_score(Y_test[:, target_column_index].reshape(-1, 1), test_predict)
    
    end_time = time()
    elapsed_time = end_time - start_time
    number = check_number_availability(randint(10000000, 99999999))
    if LOG:
        log_save(plot=[Y_train, target_column_index, Y_test, train_predict, test_predict], number=number, time=elapsed_time, model=model, scaler=scaler, epoch=EPOCH, layers_with_units=LAYERS, optimizer=OPTIMIZER, loss=LOSS, train_mse=train_mse, train_r2=train_r2,loss_eval = loss_eval ,mae = mae ,test_mse = test_mse ,test_r2 = test_r2 ,df = df)

    if not LOG:
        print(f"{time_step=}, {X.shape=}, {(len(train_predict) + time_step)=}")
        print(f"{test_predict.shape=}, {train_predict.shape=}, {scaled_data.shape=}")
        print(f"{Y_train.shape=}, {Y_test.shape=}")
        print(f"{elapsed_time=}\n{loss_eval=}\n{mae=}\n{test_mse=}\n{test_r2=}\n{train_mse=}\n{train_r2=}")
    print("--------", number, "--------", sep="\n")
        

    return 1

if '__main__' == __name__:
    print(f"URL: {URL}")
    print(f"DATA_SIZE: {DATA_SIZE}")
    print(f"EPOCH: {EPOCH}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"LAYERS: {LAYERS}")
    print(f"OPTIMIZER: {OPTIMIZER}")
    print(f"LOSS: {LOSS}")
    print(f"METRICS: {METRICS}")
    print(f"RANDOM STATE: {RANDOM}")
    print(f"LOG: {LOG}")
    run_model_train()
