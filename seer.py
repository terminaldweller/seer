#!/usr/bin/python3
# _*_ coding=utf-8 _*_

import argparse
import code
import readline
import signal
import sys
from lstm import lstm_type_1, lstm_type_2, lstm_type_3
from marionette import marrionette_type_1
from tfann import tfann_type_1

def SigHandler_SIGINT(signum, frame):
    print()
    sys.exit(0)

class Argparser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--which", type=str, help="which one to run")
        parser.add_argument("--bool", action="store_true", help="bool", default=False)
        parser.add_argument("--dbg", action="store_true", help="debug", default=False)
        self.args = parser.parse_args()

def launch_ais(which):
    if which == "marionette": marrionette_type_1()
    elif which == "lstm_type_1": lstm_type_1()
    elif which == "lstm_type_2": lstm_type_2()
    elif which == "lstm_type_3": lstm_type_3()
    elif which == "cnn_type_1": pass
    else: pass

# write code here
def premain(argparser):
    signal.signal(signal.SIGINT, SigHandler_SIGINT)
    #here
    launch_ais(argparser.args.which)

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
