#!/usr/bin/python3
# _*_ coding=utf-8 _*_
# original source-https://nicholastsmith.wordpress.com/2017/11/13/cryptocurrency-price-prediction-using-deep-learning-in-tensorflow/

#@#!pip install TFANN
import code
import readline
import signal
import sys
from TFANN import ANNR
import numpy as np
import os
import pandas as pd
import urllib.request
import matplotlib.pyplot as mpl

def GetAPIUrl(cur, sts = 1420070400):
    return 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_{:s}&start={:d}&end=9999999999&period=7200'.format(cur, sts)

def GetCurDF(cur, fp):
    openUrl = urllib.request.urlopen(GetAPIUrl(cur))
    r = openUrl.read()
    openUrl.close()
    df = pd.read_json(r.decode())
    df['date'] = df['date'].astype(np.int64) // 1000000000
    print(df.head())
    return df

class PastSampler:
    def __init__(self, N, K):
        self.K = K
        self.N = N

    def transform(self, A, Y = None):
        M = self.N + self.K     #Number of samples per row (sample + target)
        #Matrix of sample indices like: {{1, 2..., M}, {2, 3, ..., M + 1}}
        I = np.arange(M) + np.arange(A.shape[0] - M + 1).reshape(-1, 1)
        B = A[I].reshape(-1, M * A.shape[1], *A.shape[2:])
        ci = self.N * A.shape[1]    #Number of features per sample
        return B[:, :ci], B[:, ci:] #Sample matrix, Target matrix

def tfann_type_1():
    #%%Path to store cached currency data
    datPath = 'CurDat/'
    if not os.path.exists(datPath):
        os.mkdir(datPath)
    #Different cryptocurrency types
    cl = ['BTC', 'LTC', 'ETH', 'XMR']
    #Columns of price data to use
    CN = ['close', 'high', 'low', 'open', 'volume']
    #Store data frames for each of above types
    D = []
    for ci in cl:
        dfp = os.path.join(datPath, ci + '.csv')
        try:
            df = pd.read_csv(dfp, sep = ',')
        except FileNotFoundError:
            df = GetCurDF(ci, dfp)
        D.append(df)
    #%%Only keep range of data that is common to all currency types
    cr = min(Di.shape[0] for Di in D)
    for i in range(len(cl)):
        D[i] = D[i][(D[i].shape[0] - cr):]
    #%%Features are channels
    C = np.hstack((Di[CN] for Di in D))[:, None, :]
    HP = 16                 #Holdout period
    A = C[0:-HP]
    SV = A.mean(axis = 0)   #Scale vector
    C /= SV                 #Basic scaling of data
    #%%Make samples of temporal sequences of pricing data (channel)
    NPS, NFS = 256, 16         #Number of past and future samples
    ps = PastSampler(NPS, NFS)
    B, Y = ps.transform(A)
    #%%Architecture of the neural network
    NC = B.shape[2]
    #2 1-D conv layers with relu followed by 1-d conv output layer
    ns = [('C1d', [8, NC, NC * 2], 4), ('AF', 'relu'),
          ('C1d', [8, NC * 2, NC * 2], 2), ('AF', 'relu'),
          ('C1d', [8, NC * 2, NC], 2)]
    #Create the neural network in TensorFlow
    cnnr = ANNR(B[0].shape, ns, batchSize = 32, learnRate = 2e-5,
                maxIter = 64, reg = 1e-5, tol = 1e-2, verbose = True)
    cnnr.fit(B, Y)
    PTS = []                        #Predicted time sequences
    P, YH = B[[-1]], Y[[-1]]        #Most recent time sequence
    for i in range(HP // NFS):  #Repeat prediction
        P = np.concatenate([P[:, NFS:], YH], axis = 1)
        YH = cnnr.predict(P)
        PTS.append(YH)
    PTS = np.hstack(PTS).transpose((1, 0, 2))
    A = np.vstack([A, PTS]) #Combine predictions with original data
    A = np.squeeze(A) * SV  #Remove unittime dimension and rescale
    C = np.squeeze(C) * SV
    nt = 4
    PF = cnnr.PredictFull(B[:nt])
    for i in range(nt):
        fig, ax = mpl.subplots(1, 4, figsize = (16 / 1.24, 10 / 1.25))
        ax[0].plot(PF[0][i])
        ax[0].set_title('Input')
        ax[1].plot(PF[2][i])
        ax[1].set_title('Layer 1')
        ax[2].plot(PF[4][i])
        ax[2].set_title('Layer 2')
        ax[3].plot(PF[5][i])
        ax[3].set_title('Output')
        fig.text(0.5, 0.06, 'Time', ha='center')
        fig.text(0.06, 0.5, 'Activation', va='center', rotation='vertical')
        mpl.show()
    CI = list(range(C.shape[0]))
    AI = list(range(C.shape[0] + PTS.shape[0] - HP))
    NDP = PTS.shape[0] #Number of days predicted
    for i, cli in enumerate(cl):
        fig, ax = mpl.subplots(figsize = (16 / 1.5, 10 / 1.5))
        hind = i * len(CN) + CN.index('high')
        ax.plot(CI[-4 * HP:], C[-4 * HP:, hind], label = 'Actual')
        ax.plot(AI[-(NDP + 1):], A[-(NDP + 1):, hind], '--', label = 'Prediction')
        ax.legend(loc = 'upper left')
        ax.set_title(cli + ' (High)')
        ax.set_ylabel('USD')
        ax.set_xlabel('Time')
        ax.axes.xaxis.set_ticklabels([])
        mpl.show()

# write code here
def premain():
    #here
    tfann_type_1()

def main():
    premain()

if __name__ == "__main__":
    main()
