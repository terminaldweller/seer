#!/usr/bin/python3
# _*_ coding=utf-8 _*_
# original source:https://github.com/polyrabbit/hacker-news-digest/blob/master/%5Btutorial%5D%20How-to-extract-main-content-from-web-pages-using-Machine-Learning.ipynb

import argparse
import code
import readline
import signal
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC


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


# write code here
def premain(argparser):
    signal.signal(signal.SIGINT, SigHandler_SIGINT)
    # here
    dataframe = pd.read_csv("/tmp/features.csv")
    dataframe.head()
    y = dataframe.target
    X = dataframe.drop(["target"], axis=1)

    corpus = X["attr"]
    vc = CountVectorizer()
    vc.fit(corpus)

    numeric_features = pd.concat(
        [
            X.drop(["attr"], axis=1),
            pd.DataFrame(
                vc.transform(corpus).toarray(), columns=vc.vocabulary_
            ),
        ],
        axis=1,
    )
    numeric_features.head()
    plt.scatter(dataframe.index, dataframe.target, color="red", label="target")
    plt.scatter(
        numeric_features.index,
        numeric_features.depth,
        color="green",
        label="depth",
    )
    plt.scatter(
        numeric_features.index,
        numeric_features.text_ratio,
        color="blue",
        label="text_ratio",
    )
    plt.scatter(
        numeric_features.index,
        numeric_features.alink_text_ratio,
        color="skyblue",
        label="alink_text_ratio",
    )
    plt.legend(loc=(1, 0))
    plt.show()
    scaler = preprocessing.StandardScaler()
    scaler.fit(numeric_features)
    scaled_X = scaler.transform(numeric_features)

    # clf = MultinomialNB()
    # clf = RandomForestClassifier()
    clf = SVC(C=1, kernel="poly", probability=True)
    clf.fit(scaled_X, y)
    predicted_index = clf.predict(scaled_X).tolist().index(True)

    scaled_X = scaler.transform(numeric_features)
    pred_y = clf.predict(scaled_X)

    print(pd.DataFrame(clf.predict_log_proba(scaled_X), columns=clf.classes_))
    print(
        "Number of mispredicted out of %d is %d (%.2f%%)"
        % (
            y.shape[0],
            (y != pred_y).sum(),
            (y != pred_y).sum() * 100.0 / y.shape[0],
        )
    )
    print()
    print("Predicted rows:")
    print(
        dataframe[pred_y]
        .drop(["text_ratio", "alink_text_ratio", "contain_title"], axis=1)
        .merge(
            pd.DataFrame(
                clf.predict_log_proba(scaled_X)[pred_y],
                columns=clf.classes_,
                index=dataframe[pred_y].index,
            ),
            left_index=True,
            right_index=True,
        )
    )
    print()

    # print 'Acutual rows:'
    # print dataframe[dataframe.target]


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
