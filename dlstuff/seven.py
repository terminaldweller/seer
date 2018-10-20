#!/usr/bin/python3
# _*_ coding=utf-8 _*_

import argparse
import code
import readline
import signal
import sys
import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence

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

# write code here
def premain(argparser):
    signal.signal(signal.SIGINT, SigHandler_SIGINT)
    #here
    callbacks = [keras.callbacks.TensorBoard(log_dir="logfiles", histogram_freq=1, embeddings_freq=1,)]
    max_features = 2000
    max_len = 500
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    model = keras.models.Sequential()
    model.add(layers.Embedding(max_features, 128, input_length=max_len, name="embed"))
    model.add(layers.Conv1D(32, 7, activation="relu"))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 7, activation="relu"))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(1))
    summary = model.summary()
    print(summary)
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
    history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2, callbacks=callbacks)

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
