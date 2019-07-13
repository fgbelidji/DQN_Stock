import numpy as np
import pandas as pd
# Finance stuff 
from talib.abstract import SMA, RSI, ATR
from sklearn import metrics, preprocessing
from sklearn.externals import joblib

def load_data(data_filename, seq_len, test=False):
    data = pd.read_csv(data_filename)
    data.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)

    dataopen = data['open'].values
    datahigh = data['high'].values
    datalow = data['low'].values
    dataclose = data['close'].values
    datavolume = data['volume'].values
    # diff = np.diff(close)
    # diff = np.insert(diff, 0, 0)
    # sma15 = SMA(data, timeperiod=15)
    # sma60 = SMA(data, timeperiod=60)
    # rsi = RSI(data, timeperiod=14)
    # atr = ATR(data, timeperiod=14)

    # xdata = np.column_stack((close, diff, sma15, close - sma15, sma15 - sma60, rsi, atr))
    xdata = np.column_stack((dataopen, datahigh, datalow, dataclose, datavolume))
    xdata = np.nan_to_num(xdata) # format: (num_rows, 7)

    # Scaling
    if test == False:
        scaler = preprocessing.StandardScaler()
        xdata = scaler.fit_transform(xdata)
        joblib.dump(scaler, 'scaler.pkl')
    elif test == True:
        scaler = joblib.load('scaler.pkl')
        xdata = scaler.fit_transform(xdata)
    
    # # Insert timesteps for LSTM
    # result = []
    # for i in range(len(xdata) - seq_len + 1):
    #     result.append(xdata[i: i + seq_len])
    
    # result = np.array(result)

    # state_0 = result[0:1, :, :]
    # close = close[seq_len - 1:] #Price values, disregarding the first [seq_len] 

    # # Only return partial data if training
    # if test == False:
    #     row = int(round(0.9 * data.shape[0]))
    #     result = result[:row]
    #     close = close[:row]

    # return result, close, state_0

def take_action(cur_state, action, x_data, price_data, signal, time_step, eval=False):
    new_state = x_data[time_step-1:time_step, :, :]

    # Specify buy/sell signal from action
    terminal_state = 0
    if time_step + 1 == x_data.shape[0]: # if last row
        terminal_state = 1
        signal.loc[time_step] = 0
    else:
        if action == 1:
            signal_move = 100
        elif action == 2:
            signal_move = -100
        else:
            signal_move = 0
        signal.loc[time_step] = 0
    signal.fillna(value=0, inplace=True)

    # Determine reward  
    reward = 0

    return new_state, reward, terminal_state, signal
