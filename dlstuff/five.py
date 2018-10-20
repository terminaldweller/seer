#!/usr/bin/python3
# _*_ coding=utf-8 _*_

import argparse
import code
import readline
import signal
import sys
import keras
from keras import Input, Model
from keras import layers

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
    callbacks_list = [keras.callbacks.EarlyStopping(monitor="acc", patience=1,), keras.callbacks.ModelCheckpoint(filepath="mymodel.h5", monitor="val_loss", save_best_only=True,)]
    input_tensor = Input(shape=(64,))
    x = layers.Dense(32, activation="relu")(input_tensor)
    x = layers.Dense(32, activation="relu")(x)
    output_tensor = layers.Dense(10, activation="softmax")(x)
    model = Model(input_tensor, output_tensor)
    model.summary()
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])
    import numpy as np
    x_train = np.random.random((1000, 64))
    y_train = np.random.random((1000, 10))
    x_val = np.random.random((1000, 64))
    y_val = np.random.random((1000, 10))
    model.fit(x_train, y_train, epochs=10, batch_size=128, callbacks=callbacks_list, validation_data=(x_val, y_val))
    score = model.evaluate(x_train, y_train)
    print(score)
    '''
    text_vocabulary_size = 10000
    question_vocabulary_size = 10000
    answer_vocabulary_size = 500
    text_input = Input(shape=(None, ), dtype="int32", name="text")
    embedded_text = layers.Embedding(64, text_vocabulary_size)(text_input)
    encoded_text = layers.LSTM(32)(embedded_text)
    question_input = Input(shape=(None,), dtype="int32", name="question")
    embedded_question = layers.Embedding(32, question_vocabulary_size)(question_input)
    encoded_question = layers.LSTM(16)(embedded_question)
    concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)
    answer = layers.Dense(answer_vocabulary_size, activatoin="softmax")(concatenated)
    model = Model([text_input, question_input], answer)
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"])
    import numpy as np
    num_samples = 1000
    max_length = 100
    text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
    question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))
    answers = np.random.randint(0, 1, size=(num_samples, answer_vocabulary_size))
    model.fit([text, question], answers, epochs=10, batch_size=128)
    '''

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
