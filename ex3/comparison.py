import numpy as np
import models
from matplotlib import pyplot as plt


def draw_points(m):
    sample = None
    y = None

    while True:
        samples = np.random.multivariate_normal(np.zeros(2), np.eye(2), m)
        y = np.sign((np.array([0.3, -0.5]) * samples) + 0.1)
        if (-1 in y) and (1 in y):
            break
    return samples.T, y.T


def plot_hyperplanes():
    p = models.Perceptron()
    s = models.SVM()

    for i in [5, 10, 15, 25, 70]:
        X, y = draw_points(i)

        # plot the points, classified by true hypothesis:
        pos_idx = np.where(y == 1)
        neg_idx = np.where(y == -1)
        plt.plot(X[0, pos_idx], X[1, pos_idx], '.', color="blue")
        plt.plot(X[0, neg_idx], X[1, neg_idx], '.', color="orange")

        # fit:
        p.fit(X, y)
        s.fit(X, y)

        # compute predictions:
        xs = np.array(plt.gca().get_xlim())
        f = -(-0.1 + -0.3 * xs) / 0.5
        perceptron = (-p.model[1:][0] * xs - p.model[0]) / p.model[1:][1]
        svm = (-s.model[1:][0] * xs - s.model[0]) / s.model[1:][1]

        # plot predictions:
        plt.plot(xs, f, color="brown", label="True hypothesis")
        plt.plot(xs, perceptron, color="green", label="Perceptron hypothesis")
        plt.plot(xs, svm, color="pink", label="SVM hypothesis")

        # lable graph:
        plt.title("Classifications by different algorithms over m = " + str(i) + " samples")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()


def compare_classification_algorithms():
    algos = [models.Perceptron(), models.SVM(), models.LDA()]
    accuracies = np.zeros((3, 5))
    m_range = [5, 10, 15, 25, 70]

    for k, m in enumerate(m_range):
        accuracy = np.zeros((3, 500))

        for j in range(500):
            train_x, train_y = draw_points(m)
            test_x, test_y = draw_points(10000)

            for i, algo in enumerate(algos):
                algo.fit(train_x, train_y)
                accuracy[i, j] = algo.score(test_x, test_y)["accuracy"]
        accuracies[:, k] = np.mean(accuracy, axis=1)

    plt.plot(m_range, accuracies[0], label="Perceptron model")
    plt.plot(m_range, accuracies[1], label="SVM model")
    plt.plot(m_range, accuracies[2], label="LDA model")
    plt.xlabel("Number of samples")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of defferent classification models over different number of samples")
    plt.legend()
    plt.show()

