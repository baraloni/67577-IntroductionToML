import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class Classifier:

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def score(self, X, y):
        """
        Given an unlabeled test set X and the true labels y of this test set,
        returns a dictionary with the following ﬁelds:
            * num samples: number of samples in the test set
            * error: error (misclassiﬁcation) rate
            * accuracy: accuracy
            * FPR: false positive rate
            * TPR: true positive rate
            * precision: precision
            * recall: recall
        :param X:
        :param y:
        :return:
        """
        features, samples = X.shape[0] + 1, X.shape[1]
        X = np.insert(X, 0, np.ones(samples), 0)  # realizable case

        y_hat = self.predict(X)
        y_hat_pos = np.where(y_hat == 1)[0]
        y_hat_neg = np.where(y_hat == -1)[0]
        y_pos = np.where(y == 1)[0]
        y_neg = np.where(y == -1)[0]

        TP = len(np.intersect1d(y_hat_pos, y_pos))
        TN = len(np.intersect1d(y_hat_neg, y_neg))
        FP = len(np.intersect1d(y_hat_neg, y_pos))
        P = np.count_nonzero(y - 1)
        N = np.count_nonzero(y + 1)

        scores = dict()
        scores["num samples"] = X.shape[1]
        scores["error"] = np.mean(y_hat != y)
        scores["accuracy"] = (TP + TN) / (P + N)
        scores["FPR"] = FP / N
        scores["TPR"] = TP / P
        scores["precision"] = TP / (TP + FP)
        scores["recall"] = scores["TPR"]
        return scores


class Perceptron(Classifier):
    model = None

    def fit(self, X, y):
        """
        Given a training set as X and y, this method learns the parameters
         of the model and stores the trained model (namely, the variables that deﬁne
         hypothesis chosen) in self.model.
         The method returns nothing.
        :param X: shape (d,m)
        :param y: shape (m)
        :return:
        """

        features, samples = X.shape[0] + 1, X.shape[1]
        X = np.insert(X, 0, np.ones(samples), 0)  # realizable case
        w = np.zeros(features)

        while True:
            candidate_idx = np.argwhere(y * np.dot(w, X) <= 0)[0]
            print(candidate_idx)
            if len(candidate_idx) != 0:  # at least one entry holds the condition
                print(y[candidate_idx[0]])
                print(X.T[:, candidate_idx[0]])
                w += y[candidate_idx[0]] * X[:candidate_idx[0]]

            else:
                self.model = w
                break

    def predict(self, X):
        """
        Given an unlabeled test set X, predicts the label of each sample.
        Returns a vector of predicted labels y.
        :param X:
        :return:
        """
        ret = np.sign(np.dot(X, self.model[1:]) + self.model[0])
        ret[ret == 0] = 1
        return ret


class LDA(Classifier):
    mu = None
    cov = None
    pr_y = None

    def fit(self, X, y):
        samples = len(y)
        p_y = np.count_nonzero(y + 1) / samples
        self.pr_y = np.array([p_y, 1 - p_y])

        pos = np.where(y == 1)[0]
        mu_p = (np.sum(X[:, pos], axis=1) / (p_y * samples)).reshape(2, 1)
        neg = np.where(y == -1)[0]
        mu_n = (np.sum(X[:, neg], axis=1) / ((1-p_y) * samples)).reshape(2, 1)
        self.mu = np.array([mu_p, mu_n]).reshape(2, 2)

        mu_p.reshape(X[:, pos].shape)
        mu_n.reshape(X[:, neg].shape)
        sigma = (np.dot(X[:, pos] - mu_p, (X[:, pos] - mu_p).T)) + \
                (np.dot(X[:, neg] - mu_n, (X[:, neg] - mu_n).T))
        self.cov = sigma / samples

    def predict(self, X):
        y = np.array([1, -1])
        delta = []

        for i, mu in enumerate(self.mu):
            delta.append(np.dot(np.dot(X.T, np.linalg.pinv(self.cov)), mu)
                         - 0.5 * np.dot(np.dot(mu.T, np.linalg.pinv(self.cov)), mu)
                         + np.log(self.pr_y[i]))

        idx = np.argmax(np.array(delta), axis=0)
        return y[idx]


class SVM(Classifier):

    def __init__(self):
        self.model = SVC(kernel="linear", C=1e10)

    def fit(self, X, y):
        self.model.fit(X.T, y)

    def predict(self, X):
        return self.model.predict(X.T)


class Logistic(Classifier):

    def __init__(self):
        self.model = LogisticRegression(solver="liblinear")

    def fit(self, X, y):
        return self.model.fit(X.T, y)

    def predict(self, X):
        return self.model.predict(X.T)


class DecisionTree(Classifier):

    def __init__(self):
        self.model = DecisionTreeClassifier()

    def fit(self, X, y):
        return self.model.fit(X .T, y)

    def predict(self, X):
        return self.model.predict(X.T)
