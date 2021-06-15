import numpy as np
import numpy.linalg
import pandas as pd
from plotnine import *
import random


def fit_linear_regression(X, y):
    """
    Trains a linear model on the given labeled data.
    :param X: a design matrix- numpy array with p rows and n columns.
    :param y: a response vector- numpy array with n rows.
    :return: two sets of values:
        1. a numpy array of the coefficients vector ‘w‘ with p rows.
        2. a numpy array of the singular values of X.
    """
    U, s, V_T = np.linalg.svd(X, full_matrices=False)
    S_daggar_T = np.linalg.pinv(np.diag(s)).T
    x_dagger_T = np.matmul(np.matmul(U, S_daggar_T), V_T)
    w = np.matmul(x_dagger_T, y)

    return w, s

def predict(X, w):
    """
    Lets the linear model (depicted by w) predict the labels of the samples in X.
    :param X: a design matrix- numpy array with p rows and m columns.
    :param w: a numpy array of the coefficients vector ‘w‘ with p rows.
    :return: returns X's predicted labels- numpy array of m rows.
    """
    return np.dot(X, w)

def predict_log(X, w):
    """
    Lets the linear model (depicted by w) predict the labels of the samples in X.
    :param X: a design matrix- numpy array with p rows and m columns.
    :param w: a numpy array of the coefficients vector ‘w‘ with p rows.
    :return: returns X's predicted labels- numpy array of m rows.
    """
    return np.exp(np.dot(X, w))

def main():
    df = pd.read_csv("covid19_israel.csv")
    df['log_detected'] = np.log(df['detected'])
    num_of_samples = len(df['day_num'].values)
    X = df['day_num'].values.reshape(num_of_samples, 1)
    y = df['log_detected'].values.reshape(num_of_samples, 1)

    w_hat, s = fit_linear_regression(X.T, y)

    y_hat = predict(X, w_hat)
    df['prediction'] = y_hat
    print(ggplot(df) +
          geom_point(aes(x='day_num', y='log_detected'), color="black", size=0.5) +
          geom_point(aes(x='day_num', y='prediction'), color="red", shape="*", size=1.5) +
          ggtitle("In black dots: Covid19 log of detected over the num of days.\n"
                  "In red stars: Model predictions."))
    df = df.drop(['prediction'], 1)

    y_hat = predict_log(X, w_hat)
    df['prediction'] = y_hat
    print(ggplot(df) +
          geom_point(aes(x='day_num', y='detected'), color="black", size=0.5) +
          geom_point(aes(x='day_num', y='prediction'), color="red", shape="*", size=1.5) +
          ggtitle("In black dots: Covid19 number of detected over the num of days.\n" +
                  "In red stars: Model predictions."))

    df = df.drop(['prediction'], 1)

