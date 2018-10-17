#!/usr/bin/python3
# _*_ coding=utf-8 _*_

import argparse
import code
import readline
import signal
import sys
import numpy as np
from keras.datasets import reuters
from keras import models
from keras import layers
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

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

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(sequences), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

def plot_loss(history):
    history_dic = history.history
    loss_values = history_dic["loss"]
    val_loss_values = history_dic["val_loss"]
    epochs = range(1, len(history_dic["loss"]) + 1)
    plt.plot(epochs, loss_values, "bo", label="Training Loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation Loss")
    plt.title("training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def plot_acc(history):
    history_dic = history.history
    acc_values = history_dic["acc"]
    val_acc_values = history_dic["val_acc"]
    epochs = range(1, len(history_dic["acc"]) + 1)
    plt.plot(epochs, acc_values, "bo", label="Training Acc")
    plt.plot(epochs, val_acc_values, "b", label="Validation Acc")
    plt.title("training and validation acc")
    plt.xlabel("Epochs")
    plt.ylabel("Acc")
    plt.legend()
    plt.show()

# write code here
def premain(argparser):
    signal.signal(signal.SIGINT, SigHandler_SIGINT)
    #here
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
    #print(len(train_data))
    #print(len(test_data))
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    #one_hot_train_labels = to_one_hot(train_labels)
    #one_hot_test_labels = to_one_hot(test_labels)
    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels)

    model = models.Sequential()
    model.add(layers.Dense(64, activation="relu", input_shape=(10000,)))
    #model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(46, activation="softmax"))
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]
    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]
    history = model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, validation_data=(x_val, y_val))
    '''
    plot_loss(history)
    plt.clf()
    plot_acc(history)
    '''
    results = model.evaluate(x_test, one_hot_test_labels)
    print(results)

    predictions = model.predict(x_test)



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
