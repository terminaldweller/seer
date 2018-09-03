#!/usr/bin/python3
# _*_ coding=utf-8 _*_
# original source-https://medium.com/@huangkh19951228/predicting-cryptocurrency-price-with-tensorflow-and-keras-e1674b0dc58a

import argparse
import code
import readline
import signal
import sys
import json
import numpy as np
import os
import pandas as pd
import urllib3
import requests
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv1D, MaxPooling1D, LeakyReLU, PReLU
from keras.layers import LSTM
from keras.layers import GRU, CuDNNGRU
from keras.utils import np_utils
from keras.callbacks import CSVLogger, ModelCheckpoint
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import applications
from keras.models import Model
from scipy.ndimage import imread
import random
from keras import backend as K
import keras
from keras import optimizers
import matplotlib.pyplot as plt

def SigHandler_SIGINT(signum, frame):
    print()
    sys.exit(0)

class Argparser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--string", type=str, help="string")
        parser.add_argument("--bool", action="store_true", help="bool", default=False)
        parser.add_argument("--dbg", action="store_true", help="debug", default=False)
        self.args = parser.parse_args()

class PastSampler(object):
    def __init__(self, N, K, sliding_window=True):
        self.N = N
        self.K = K
        self.sliding_window = sliding_window

    def transform(self, A):
        M = self.N + self.K
        if self.sliding_window:
            I = np.arange(M) + np.arange(A.shape[0] - M + 1).reshape(-1,1)
        else:
            if A.shape[0]%M == 0:
                I = np.arange(M) + np.arange(0, A.shape[0], M).reshape(-1,1)
            else:
                I = np.arange(M) + np.arange(0, A.shape[0] - M, M).reshape(-1,1)

        B = A[I].reshape(-1, M*A.shape[1], A.shape[2])
        ci = self.N*A.shape[1]
        return B[:, :ci], B[:, ci:]

def getData(symbol_str):
    data_file = Path("./cnn/" + symbol_str + ".csv")
    original_columns =["close", "date", "high", "low", "open"]
    new_columns = ["Close", "Timestamp", "High", "Low", "Open"]
    columns = ["Close"]
    url = "https://poloniex.com/public?command=returnChartData&currencyPair=USDT_" + symbol_str + "&start=1356998100&end=9999999999&period=300"
    r = requests.get(url)
    d = json.loads(r.content.decode("utf-8"))
    df = pd.DataFrame(d)

    df = df.loc[:, original_columns]
    df.columns = new_columns
    df.to_csv("./cnn/" + symbol_str + ".csv", index=None)
    df = pd.read_csv(data_file)
    time_stamps = df["Timestamp"]
    df = df.loc[:, columns]
    original_df = pd.read_csv(data_file).loc[:, columns]
    return df, original_df, time_stamps

def Scaler(df, original_df, time_stamps, symbol_str):
    file_name="./cnn/" + symbol_str + "_close.h5"
    scaler = MinMaxScaler()
    columns= ["Close"]
    for c in columns:
        df[c] = scaler.fit_transform(df[c].values.reshape(-1,1))
    A = np.array(df)[:,None,:]
    original_A = np.array(original_df)[:,None,:]
    time_stamps = np.array(time_stamps)[:,None,None]
    NPS, NFS = 256, 16
    ps = PastSampler(NPS, NFS, sliding_window=False)
    B, Y = ps.transform(A)
    input_times, output_times = ps.transform(time_stamps)
    original_B, original_Y = ps.transform(original_A)

    with h5py.File(file_name, "w") as f:
        f.create_dataset("inputs", data=B)
        f.create_dataset("outputs", data=Y)
        f.create_dataset("input_times", data=input_times)
        f.create_dataset("output_times", data=output_times)
        f.create_dataset("original_datas", data=np.array(original_df))
        f.create_dataset("original_inputs", data=original_B)
        f.create_dataset("original_outputs", data=original_Y)

def cnn_type_1(symbol_str):
    df, original_df, time_stamps = getData(symbol_str)
    Scaler(df, original_df, time_stamps, symbol_str)
    # run on gpu
    '''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    '''

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    with h5py.File("".join("./cnn/" + symbol_str + "_close.h5"), "r") as hf:
        datas = hf["inputs"].value
        labels = hf["outputs"].value

    output_file_name = "cnn/" + symbol_str + "_CNN_2_relu"
    step_size = datas.shape[1]
    batch_size = 8
    nb_features = datas.shape[2]

    epochs = 100

    #split training validation
    training_size = int(0.8* datas.shape[0])
    training_datas = datas[:training_size,:]
    training_labels = labels[:training_size,:]
    validation_datas = datas[training_size:,:]
    validation_labels = labels[training_size:,:]

    model = Sequential()

    # 2 Layers
    model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=3, filters=8, kernel_size=20))
    model.add(Dropout(0.5))
    model.add(Conv1D( strides=4, filters=nb_features, kernel_size=16))

    '''
    # 3 Layers
    model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=3, filters=8, kernel_size=8))
    #model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Conv1D(activation='relu', strides=2, filters=8, kernel_size=8))
    #model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Conv1D( strides=2, filters=nb_features, kernel_size=8))
    # 4 layers
    model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=2, filters=8, kernel_size=2))
    #model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Conv1D(activation='relu', strides=2, filters=8, kernel_size=2))
    #model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Conv1D(activation='relu', strides=2, filters=8, kernel_size=2))
    #model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Conv1D( strides=2, filters=nb_features, kernel_size=2))
    '''

    model.compile(loss='mse', optimizer='adam')
    model.fit(training_datas, training_labels,verbose=1, batch_size=batch_size,validation_data=(validation_datas,validation_labels), epochs = epochs, callbacks=[CSVLogger(output_file_name+'.csv', append=True),ModelCheckpoint(output_file_name+'-{epoch:02d}-{val_loss:.5f}.hdf5', monitor='val_loss', verbose=1,mode='min')])

def lstm_type_cnn_1(symbol_str, kind):
    df, original_df, time_stamps = getData(symbol_str)
    Scaler(df, original_df, time_stamps, symbol_str)
    """
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    """

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    with h5py.File("".join("./cnn/" + symbol_str + "_close.h5"), "r") as hf:
        datas = hf['inputs'].value
        labels = hf['outputs'].value

    step_size = datas.shape[1]
    units= 50
    second_units = 30
    batch_size = 8
    nb_features = datas.shape[2]
    epochs = 100
    output_size=16
    output_file_name = "cnn/" + symbol_str + "_CNN_LSTM_2_relu"
    #split training validation
    training_size = int(0.8* datas.shape[0])
    training_datas = datas[:training_size,:]
    training_labels = labels[:training_size,:,0]
    validation_datas = datas[training_size:,:]
    validation_labels = labels[training_size:,:,0]

    #build model
    model = Sequential()
    if kind == "GRU":
        model.add(GRU(units=units,activation='tanh', input_shape=(step_size,nb_features),return_sequences=False))
    elif kind == "LSTM":
        model.add(LSTM(units=units,activation='tanh', input_shape=(step_size,nb_features),return_sequences=False))
    model.add(Dropout(0.8))
    model.add(Dense(output_size))
    model.add(LeakyReLU())
    model.compile(loss='mse', optimizer='adam')
    model.fit(training_datas, training_labels, batch_size=batch_size,validation_data=(validation_datas,validation_labels), epochs = epochs, callbacks=[CSVLogger(output_file_name+'.csv', append=True),ModelCheckpoint(output_file_name+'-{epoch:02d}-{val_loss:.5f}.hdf5', monitor='val_loss', verbose=1,mode='min')])

def load_cnn_type_1(symbol_str, vis_year, vis_month):
    df, original_df, time_stamps = getData(symbol_str)
    Scaler(df, original_df, time_stamps, symbol_str)
    """
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    """

    with h5py.File("".join("./cnn/" + symbol_str + "_close.h5"), "r") as hf:
        datas = hf['inputs'].value
        labels = hf['outputs'].value
        input_times = hf['input_times'].value
        output_times = hf['output_times'].value
        original_inputs = hf['original_inputs'].value
        original_outputs = hf['original_outputs'].value
        original_datas = hf['original_datas'].value

    scaler=MinMaxScaler()
    #split training validation
    training_size = int(0.8* datas.shape[0])
    training_datas = datas[:training_size,:,:]
    training_labels = labels[:training_size,:,:]
    validation_datas = datas[training_size:,:,:]
    validation_labels = labels[training_size:,:,:]
    validation_original_outputs = original_outputs[training_size:,:,:]
    validation_original_inputs = original_inputs[training_size:,:,:]
    validation_input_times = input_times[training_size:,:,:]
    validation_output_times = output_times[training_size:,:,:]

    ground_true = np.append(validation_original_inputs,validation_original_outputs, axis=1)
    ground_true_times = np.append(validation_input_times,validation_output_times, axis=1)
    step_size = datas.shape[1]
    batch_size= 8
    nb_features = datas.shape[2]

    model = Sequential()

    # 2 layers
    model.add(Conv1D(activation='relu', input_shape=(step_size, nb_features), strides=3, filters=8, kernel_size=20))
    # model.add(LeakyReLU())
    model.add(Dropout(0.25))
    model.add(Conv1D( strides=4, filters=nb_features, kernel_size=16))
    model.load_weights("cnn/" + symbol_str + "_CNN_2_relu-76-0.00036.hdf5")
    model.compile(loss='mse', optimizer='adam')

    predicted = model.predict(validation_datas)
    predicted_inverted = []

    for i in range(original_datas.shape[1]):
        scaler.fit(original_datas[:, i].reshape(-1, 1))
        predicted_inverted.append(scaler.inverse_transform(predicted[:,:,i]))
    print(np.array(predicted_inverted).shape)
    #get only the close data
    ground_true = ground_true[:,:,0].reshape(-1)
    ground_true_times = ground_true_times.reshape(-1)
    ground_true_times = pd.to_datetime(ground_true_times, unit='s')
    # since we are appending in the first dimension
    predicted_inverted = np.array(predicted_inverted)[0,:,:].reshape(-1)
    print(np.array(predicted_inverted).shape)
    validation_output_times = pd.to_datetime(validation_output_times.reshape(-1), unit='s')

    ground_true_df = pd.DataFrame()
    ground_true_df['times'] = ground_true_times
    ground_true_df['value'] = ground_true

    prediction_df = pd.DataFrame()
    prediction_df['times'] = validation_output_times
    prediction_df['value'] = predicted_inverted

    prediction_df = prediction_df.loc[(prediction_df["times"].dt.year == vis_year )&(prediction_df["times"].dt.month > vis_month ),: ]
    ground_true_df = ground_true_df.loc[(ground_true_df["times"].dt.year == vis_year )&(ground_true_df["times"].dt.month > vis_month ),:]

    plt.figure(figsize=(20,10))
    plt.plot(ground_true_df.times,ground_true_df.value, label = 'Actual')
    plt.plot(prediction_df.times,prediction_df.value,'ro', label='Predicted')
    plt.legend(loc='upper left')
    plt.show()

# write code here
def premain(argparser):
    signal.signal(signal.SIGINT, SigHandler_SIGINT)
    #here
    #cnn_type_1("ETH")
    #lstm_type_cnn_1("ETH", "GRU")
    load_cnn_type_1("ETH", 2018, 4)

def main():
    argparser = Argparser()
    if argparser.args.dbg:
        try:
            premain(argparser)
        except Exception as e:
            print(e.__doc__)
            if e.message: print(e.message)
            variables = globals().copy()
            variables.update(locals())
            shell = code.InteractiveConsole(variables)
            shell.interact(banner="DEBUG REPL")
    else:
        premain(argparser)

if __name__ == "__main__":
    main()
