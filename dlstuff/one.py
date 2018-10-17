#!/usr/bin/python3
# _*_ coding=utf-8 _*_

import argparse
import code
import readline
import signal
import sys
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
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

# write code here
def premain(argparser):
    signal.signal(signal.SIGINT, SigHandler_SIGINT)
    #here
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    '''
    print(train_images.shape)
    print(len(train_labels))
    print(train_labels)
    print(test_images.shape)
    print(len(test_labels))
    print(test_labels)
    digit = train_images[4]
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()
    '''

    network = models.Sequential()
    network.add(layers.Dense(512, activation="relu", input_shape=(28*28,)))
    network.add(layers.Dense(10, activation="softmax"))
    #network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    network.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype("float32") / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    network.fit(train_images, train_labels, epochs=5, batch_size=128)

    test_loss, test_acc = network.evaluate(test_images, test_labels)
    print("test_acc:", test_acc)

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
