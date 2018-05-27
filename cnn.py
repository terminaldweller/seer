#!/usr/bin/python3

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
            if A.shapep[0]%M == 0:
                I = np.arange(M) + np.arange(0, A.shape[0], M).reshape(-1,1)
            else:
                I = np.arange(M) + np.arange(0, A.shape[0] - M, M).reshape(-1,1)

        B = A[I].reshape(-1, M*A.shape[1], A.shape[2])
        ci = self.N*A.shape[1]
        return B[:, :ci], B[:, ci:]

def getData(symbol_str):
    data_file = Path("./" + symbol_str + ".csv")
    original_columns =["close", "date", "high", "low", "open"]
    new_columns = ["Close", "Timestamp", "High", "Low", "Open"]
    columns = ["Close"]
    if data_file.is_file():
        original_data_file = pd.read_csv(data_file).loc[:, columns]
        return pd.read_csv(data_file).loc[:, columns], original_data_file
    else:
        url = "https://poloniex.com/public?command=returnChartData&currencyPair=USDT_" + symbol_str + "&start=1356998100&end=9999999999&period=300"
        r = requests.get(url)
        d = json.loads(r.content.decode("utf-8"))
        df = pd.DataFrame(d)

        df = df.loc[:, original_columns]
        df.columns = new_columns
        df.to_csv(symbol_str + ".csv", index=None)
        df = pd.read_csv(data_file)
        time_stamps = df["Timestamp"]
        df = df.loc[:, columns]
        original_data_file = pd.read_csv(data_file).loc[:, columns]
        return df

# write code here
def premain(argparser):
    signal.signal(signal.SIGINT, SigHandler_SIGINT)
    #here
    columns = ["Close"]
    btc_df, orig_btc = getData("BTC")
    eth_df, orig_eth = getData("ETH")
    scaler = MinMaxScaler()
    for c in columns:
        btc_df[c] = scaler.fit_transform(btc_df[c].values.reshape(-1, 1))
        eth_df[c] = scaler.fit_transform(eth_df[c].values.reshape(-1, 1))

    A = np.array(eth_df)[:,None,:]
    original_A = np.array(orig_eth)[:,None,:]
    time_stamps = np.array(time_stamps)[:, None, None]

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
