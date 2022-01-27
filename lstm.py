#!python
# _*_ coding=utf-8 _*_
# original source:https://github.com/dashee87/blogScripts/blob/master/Jupyter/2017-11-20-predicting-cryptocurrency-prices-with-deep-learning.ipynb

# @#!pip install lxml
# @#!mkdir lstm-models
import argparse
import code
import readline
import signal
import sys
import pandas as pd
import json
import os
import numpy as np
import urllib3
import time
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model
from keras import models
from keras import layers

window_len = 10
split_date = "2018-03-01"


def SigHandler_SIGINT(signum, frame):
    print()
    sys.exit(0)


class Argparser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--string", type=str, help="string")
        parser.add_argument(
            "--bool", action="store_true", help="bool", default=False
        )
        parser.add_argument(
            "--dbg", action="store_true", help="debug", default=False
        )
        self.args = parser.parse_args()


def getData_CMC(crypto, crypto_short):
    market_info = pd.read_html(
        "https://coinmarketcap.com/currencies/"
        + crypto
        + "/historical-data/?start=20160428&end="
        + time.strftime("%Y%m%d")
    )[0]
    print(type(market_info))
    market_info = market_info.assign(Date=pd.to_datetime(market_info["Date"]))
    # print(market_info)
    # if crypto == "ethereum": market_info.loc[market_info["Market Cap"]=="-","Market Cap"]=0
    # if crypto == "dogecoin": market_info.loc[market_info["Volume"]=="-","Volume"]=0
    market_info["Volume"] = market_info["Volume"].astype("int64")
    market_info.columns = market_info.columns.str.replace("*", "")
    # print(type(market_info))
    # print(crypto + " head: ")
    # print(market_info.head())
    kwargs = {
        "close_off_high": lambda x: 2
        * (x["High"] - x["Close"])
        / (x["High"] - x["Low"])
        - 1,
        "volatility": lambda x: (x["High"] - x["Low"]) / (x["Open"]),
    }
    market_info = market_info.assign(**kwargs)
    model_data = market_info[
        ["Date"]
        + [
            coin + metric
            for coin in [""]
            for metric in ["Close", "Volume", "close_off_high", "volatility"]
        ]
    ]
    model_data = model_data.sort_values(by="Date")
    # print(model_data.head())
    print(type(model_data))
    return model_data


def getData_Stock(name, period):
    info = pd.read_csv(
        "./data/" + name + "/" + period + ".csv", encoding="utf-16"
    )
    return info


def get_sets(crypto, model_data):
    training_set, test_set = (
        model_data[model_data["Date"] < split_date],
        model_data[model_data["Date"] >= split_date],
    )
    training_set = training_set.drop("Date", 1)
    test_set = test_set.drop("Date", 1)
    norm_cols = [
        coin + metric for coin in [] for metric in ["Close", "Volume"]
    ]
    LSTM_training_inputs = []
    for i in range(len(training_set) - window_len):
        temp_set = training_set[i : (i + window_len)].copy()
        for col in norm_cols:
            temp_set.loc[:, col] = temp_set[col] / temp_set[col].iloc[0] - 1
        LSTM_training_inputs.append(temp_set)
    LSTM_training_outputs = (
        training_set["Close"][window_len:].values
        / training_set["Close"][:-window_len].values
    ) - 1
    LSTM_test_inputs = []
    for i in range(len(test_set) - window_len):
        temp_set = test_set[i : (i + window_len)].copy()
        for col in norm_cols:
            temp_set.loc[:, col] = temp_set[col] / temp_set[col].iloc[0] - 1
        LSTM_test_inputs.append(temp_set)
    LSTM_test_outputs = (
        test_set["Close"][window_len:].values
        / test_set["Close"][:-window_len].values
    ) - 1
    print(LSTM_training_inputs[0])
    LSTM_training_inputs = [
        np.array(LSTM_training_input)
        for LSTM_training_input in LSTM_training_inputs
    ]
    LSTM_training_inputs = np.array(LSTM_training_inputs)

    LSTM_test_inputs = [
        np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs
    ]
    LSTM_test_inputs = np.array(LSTM_test_inputs)
    return LSTM_training_inputs, LSTM_test_inputs, training_set, test_set


def build_model(
    inputs,
    output_size,
    neurons,
    activ_func="linear",
    dropout=0.25,
    loss="mae",
    optimizer="adam",
):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model


def stock():
    split_date = "2017.01.01"
    model_data = getData_Stock("irxo", "Daily")
    model_data = model_data.sort_values(by="Date")

    training_set, test_set = (
        model_data[model_data["Date"] < split_date],
        model_data[model_data["Date"] >= split_date],
    )
    training_set = training_set.drop("Date", 1)
    test_set = test_set.drop("Date", 1)

    training_inputs = training_set
    training_outputs = training_set.drop(
        ["Open", "High", "Low", "NTx", "Volume"], axis=1
    )
    test_inputs = test_set
    test_outputs = test_set.drop(
        ["Open", "High", "Low", "NTx", "Volume"], axis=1
    )

    print(training_set.head)
    print(test_set.head)
    print(training_inputs.shape)
    print(test_inputs.shape)
    print(training_outputs.shape)
    print(test_outputs.shape)

    model = models.Sequential()
    model.add(
        layers.Dense(
            64, activation="relu", input_shape=(training_inputs.shape[1],)
        )
    )
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1))
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    history = model.fit(
        training_inputs,
        training_outputs,
        validation_data=(test_inputs, test_outputs),
        epochs=10,
        batch_size=1,
        verbose=2,
    )


def lstm_type_1(crypto, crypto_short):
    model_data = getData_CMC(crypto, crypto_short)
    np.random.seed(202)
    training_inputs, test_inputs, training_set, test_set = get_sets(
        crypto, model_data
    )
    model = build_model(training_inputs, output_size=1, neurons=20, loss="mse")
    training_outputs = (
        training_set["Close"][window_len:].values
        / training_set["Close"][:-window_len].values
    ) - 1
    history = model.fit(
        training_inputs,
        training_outputs,
        epochs=50,
        batch_size=1,
        verbose=2,
        shuffle=True,
    )


def lstm_type_4(crypto, crypto_short, crypto2, crypto_short2):
    model_data = getData_CMC(crypto, crypto_short)
    model_data2 = getData_CMC(crypto2, crypto_short2)
    np.random.seed(202)
    training_inputs, test_inputs, training_set, test_set = get_sets(
        crypto, model_data
    )
    training_inputs2, test_inputs2, training_set2, test_set2 = get_sets(
        crypto2, model_data2
    )
    return
    model = build_model(
        training_inputs / training_inputs2,
        output_size=1,
        neurons=20,
        loss="mse",
    )
    training_outputs = (
        (training_set["Close"][window_len:].values)
        / (training_set["Close"][:-window_len].values)
    ) - 1
    history = model.fit(
        training_inputs / training_inputs2,
        training_outputs,
        epochs=10,
        batch_size=1,
        verbose=2,
        shuffle=True,
    )


def lstm_type_2(crypto, crypto_short, pred_range, neuron_count):
    model_data = getData_CMC(crypto, crypto_short)
    np.random.seed(202)
    training_inputs, test_inputs, training_set, test_set = get_sets(
        crypto, model_data
    )
    model = build_model(
        training_inputs,
        output_size=pred_range,
        neurons=neuron_count,
        loss="mse",
    )
    training_outputs = (
        training_set["Close"][window_len:].values
        / training_set["Close"][:-window_len].values
    ) - 1
    training_outputs = []
    for i in range(window_len, len(training_set["Close"]) - pred_range):
        training_outputs.append(
            (
                training_set["Close"][i : i + pred_range].values
                / training_set["Close"].values[i - window_len]
            )
            - 1
        )
    training_outputs = np.array(training_outputs)
    history = model.fit(
        training_inputs[:-pred_range],
        training_outputs,
        epochs=50,
        batch_size=1,
        verbose=2,
        shuffle=True,
    )


def lstm_type_3(crypto, crypto_short, pred_range, neuron_count):
    model_data = getData_CMC(crypto, crypto_short)
    np.random.seed(202)
    training_inputs, test_inputs, training_set, test_set = get_sets(
        crypto, model_data
    )
    model = build_model(training_inputs, output_size=1, neurons=neuron_count)
    training_outputs = (
        training_set["Close"][window_len:].values
        / training_set["Close"][:-window_len].values
    ) - 1
    training_outputs = []
    for rand_seed in range(775, 800):
        print(rand_seed)
        np.random.seed(rand_seed)
        temp_model = build_model(
            training_inputs, output_size=1, neurons=neuron_count
        )
        temp_model.fit(
            training_inputs,
            (
                training_set["Close"][window_len:].values
                / training_set["Close"][:-window_len].values
            )
            - 1,
            epochs=50,
            batch_size=1,
            verbose=0,
            shuffle=True,
        )
        temp_model.save(
            "./lstm-models/" + crypto + "_model_randseed_%d.h5" % rand_seed
        )


def load_models(crypto, crypto_short):
    preds = []
    model_data = getData_CMC(crypto, crypto_short)
    np.random.seed(202)
    training_inputs, test_inputs, training_set, test_set = get_sets(
        crypto, model_data
    )
    for rand_seed in range(775, 800):
        temp_model = load_model(
            "./lstm-models/" + crypto + "_model_randseed_%d.h5" % rand_seed
        )
        preds.append(
            np.mean(
                abs(
                    np.transpose(temp_model.predict(test_inputs))
                    - (
                        test_set["Close"].values[window_len:]
                        / test_set["Close"].values[:-window_len]
                        - 1
                    )
                )
            )
        )


# write code here
def premain(argparser):
    signal.signal(signal.SIGINT, SigHandler_SIGINT)
    # here
    # lstm_type_1("ethereum", "ether")
    # lstm_type_2("ethereum", "ether", 5, 20)
    # lstm_type_3("ethereum", "ether", 5, 20)
    # lstm_type_4("ethereum", "ether", "dogecoin", "doge")
    # load_models("ethereum", "eth")
    stock()


def main():
    argparser = Argparser()
    if argparser.args.dbg:
        try:
            premain(argparser)
        except Exception as e:
            print(e.__doc__)
            if e.message:
                print(e.message)
            variables = globals().copy()
            variables.update(locals())
            shell = code.InteractiveConsole(variables)
            shell.interact(banner="DEBUG REPL")
    else:
        premain(argparser)


if __name__ == "__main__":
    main()
