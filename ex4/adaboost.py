"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Gad Zalcberg
Date: February, 2019

"""
import numpy as np


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights   
      
    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        num_samples = len(y)
    
        # initialization:
        D = np.zeros(shape=(self.T + 1, num_samples))
        D[0] = np.ones(shape=num_samples) / num_samples

        for t in range(self.T):
            curr_D = D[t]
            # fit weak learner (it does so in init)
            self.h[t] = self.WL(curr_D, X, y)

            # calculate error and weight from weak learner prediction
            y_hat = self.h[t].predict(X)
            err_t = curr_D[(y_hat != y)].sum()
            self.w[t] = np.log((1 - err_t) / err_t) / 2

            # update sample weights and normalize them:
            D[t + 1] = curr_D * np.exp(-self.w[t] * y * y_hat)
            D[t + 1] /= D[t + 1].sum()

        return D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        return np.sign(np.zeros(shape=(len(X))) + (self.w[:max_t] @ [self.h[t].predict(X) for t in range(max_t)]))

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the wrong predictions when predict only with max_t weak learners (float)
        """ 
        return np.ones(len(y))[(self.predict(X, max_t) != y)].sum() / len(y)

