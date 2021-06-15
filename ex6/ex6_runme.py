import SVM
import load_data as ld
import numpy as np
from plotnine import *
import pandas as pd

x_train, y_train, x_test, y_test = ld.get_digits(0, 1)

m, d = x_test.shape
y_train = 2 * y_train - 1  # change labels from {0,1} to {-1, 1}
y_test = 2 * y_test - 1  # change labels from {0,1} to {-1, 1}

gamma = np.random.rand(1)

batches = [5, 50, 100]
iters = 150

eta = 0.001
w = np.random.rand(d + 1, 1)

# GD:
gd_ret = gd(x_train, y_train, iters, eta, w, gamma)

# SGD:
sgd_rets = []
for batch in batches:
    sgd_ret = sgd(x_train, y_train, iters, eta, w, batch, gamma)
    sgd_rets.append(sgd_ret)


# plot:


def plot():
    dfs = []
    dfs.append(pd.DataFrame(
        {"lost": [test_error(w, x_test, y_test) for w in gd_ret], "iterations": range(iters + 1), "algo": "gd"}))
    dfs.append(pd.DataFrame(
        {"lost": [test_error(w, x_test, y_test) for w in sgd_rets[0]], "iterations": range(iters + 1),
         "algo": "sgd-5"}))
    dfs.append(pd.DataFrame(
        {"lost": [test_error(w, x_test, y_test) for w in sgd_rets[1]], "iterations": range(iters + 1),
         "algo": "sgd-50"}))
    dfs.append(pd.DataFrame(
        {"lost": [test_error(w, x_test, y_test) for w in sgd_rets[2]], "iterations": range(iters + 1),
         "algo": "sgd-100"}))
    df = pd.concat(dfs)

    p = (ggplot(df, aes(y='lost', x='iterations', color='algo')) +
         geom_point(size=2) + geom_line() +
         labs(x="iterations", y="lost", title=r"Lost over number of iterations"))
    return p


plot()