#!/usr/bin/python3

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

def getData_CMC(crypto):
    coin_market_info = pd.read_html("https://coinmarketcap.com/currencies/"+crypto+"/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]
    coin_market_info =  coin_market_info.assign(Date=pd.to_datetime(coin_market_info['Date']))
    #new_list = list(coin_market_info.keys())
    #print(repr(new_list))
    #for k,v in coin_market_info.items():
        #print(repr(k) + " : " + repr(v))
    if crypto == "ethereum": coin_market_info.loc[coin_market_info["Market Cap"]=="-","Market Cap"]=0
    if crypto == "dogecoin": coin_market_info.loc[coin_market_info["Volume"]=="-","Volume"]=0
    #coin_market_info.loc[coin_market_info['High']=="-",'High']=0
    #coin_market_info.loc[coin_market_info['Low']=="-",'Low']=0
    #coin_market_info.loc[coin_market_info['Open']=="-",'Open']=0
    #coin_market_info.loc[coin_market_info['Close']=="-",'Close']=0
    print(crypto + " head: ")
    print(coin_market_info.head())
    #print(repr(coin_market_info))
    return coin_market_info

def build_model(inputs, output_size, neurons, activ_func="linear", dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model

# write code here
def premain(argparser):
    signal.signal(signal.SIGINT, SigHandler_SIGINT)
    #here
    #getData_CMC("bitcoin")
    eth_data = getData_CMC("ethereum")
    doge_data = getData_CMC("dogecoin")
    np.random.seed(202)
    eth_model = build_model(LSTM_training_inputs, output_size=1, neurons=20)

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
