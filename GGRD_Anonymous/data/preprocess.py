import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

def getTimeEmbedding(time):
    df = pd.DataFrame(time, columns=['time'])
    df['time'] = pd.to_datetime(df['time'])

    df['minute'] = df['time'].apply(lambda row: row.minute / 59 - 0.5)
    df['hour'] = df['time'].apply(lambda row: row.hour / 23 - 0.5)
    df['weekday'] = df['time'].apply(lambda row: row.weekday() / 6 - 0.5)
    df['day'] = df['time'].apply(lambda row: row.day / 30 - 0.5)
    df['month'] = df['time'].apply(lambda row: row.month / 11 - 0.5)

    return df[['minute', 'hour', 'weekday', 'day', 'month']].values

def getData(path='./dataset/', dataset='SWaT',  train_rate=0.8):
    init_data = np.load(path + dataset + '/' + dataset + '_train_data.npy')  
    test_data = np.load(path + dataset + '/' + dataset + '_test_data.npy')
    init_time = getTimeEmbedding(np.load(path + dataset + '/' + dataset + '_train_date.npy'))
    test_time = getTimeEmbedding(np.load(path + dataset + '/' + dataset + '_test_date.npy'))

    test_label = np.load(path + dataset + '/' + dataset + '_test_label.npy')

    scaler = StandardScaler()
    scaler.fit(init_data)
    init_data = pd.DataFrame(scaler.transform(init_data)).fillna(0).values
    test_data = pd.DataFrame(scaler.transform(test_data)).fillna(0).values

    init_label = np.zeros((len(init_data), 1))

    train_data = init_data[:int(train_rate * len(init_data)), :]
    train_time = init_time[:int(train_rate * len(init_time)), :]
    train_label = init_label[:int(train_rate * len(init_label)), :]

    valid_data = init_data[int(train_rate * len(init_data)):, :]
    valid_time = init_time[int(train_rate * len(init_time)):, :]
    valid_label = init_label[int(train_rate * len(init_label)):, :]

    data = {
        'train_data': train_data, 'train_time': train_time,  'train_label': train_label,
        'valid_data': valid_data, 'valid_time': valid_time,  'valid_label': valid_label,
        'init_data': init_data, 'init_time': init_time,  'init_label': init_label,
        'test_data': test_data, 'test_time': test_time,  'test_label': test_label
    }

    return data
