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


def mse(y, y_hat):
    """
    :param y: a response vector- numpy array of n rows.
    :param y_hat: a prediction vector- numpy array of n rows.
    :return: the MSE over the received samples (min squared loss between y and y_hat)
    """
    ys_squared_diff = np.linalg.norm(y_hat - y) **2
    return len(y) * ys_squared_diff


def preprocess_data(df):
    """
    preprocess the data: remove rows that contains invalid data or data that doesn't make sense.
    :param df: the df to pre-process
    :return: the pre processed data as dataset
    """
    #  filter out coulumns that contains, and shouldn't, non-positive values:
    df = df[df.price > 0]
    df = df[(df.bedrooms > 0) & (df.bedrooms < 33)]  # there is an  house with 33 bedrooms and 1.75 bathrooms, probably a typing mistake
    df = df[df.bathrooms > 0]
    df = df[df.sqft_living > 0]
    df = df[df.sqft_lot > 0]
    df = df[df.floors > 0]
    df = df[df.sqft_above > 0]
    df = df[df.yr_built > 0]
    df = df[df.yr_renovated > 0]
    df = df[df.sqft_living15 > 0]
    df = df[df.sqft_lot15 > 0]

    #  filter out columns that do not apply to common sense:
    df = df[(df.yr_renovated > df.yr_built)]  # built before renovated
    df = df[(df.sqft_above + df.sqft_basement) == df.sqft_living]  # living area needs to add up
    df = df[(df.zipcode < 98200) & (df.zipcode > 98000)]  # legal kings county zip code

    # dummies- zipcode:
    df = pd.get_dummies(data=df, columns=['zipcode'],  drop_first=True)  # TODO- true?

    df = df.drop_duplicates('id')  # remove duplicated id's

    # remove uninteresting columns:
    df = df.drop(['date'], 1)
    df = df.drop(['id'], 1)

    return df


def load_data(path):
    """
    loads the csv file in path to a dataset and preprocess
    it to a form of a valid design matrix.
    :param path: a valid path to a sv file
    :return: the pre processed data:
            X: the preprocessed design matrix as a dataset.
            y: the response vector.
    """
    df = pd.read_csv(path)
    processed_data = preprocess_data(df.dropna())
    processed_data['homogeneous'] = [1] * len(processed_data)

    y = processed_data['price']
    processed_data = processed_data.drop(['price'], 1)
    return processed_data.values, y.values


def plot_singular_values(singular_values):
    """

    :param singular_values:
    :return:
    """
    idx_range = numpy.arange(len(singular_values))
    singular_values = sorted(singular_values)[::-1]

    df = pd.DataFrame({'index': idx_range, 'singular values': singular_values})
    print(ggplot(df, aes(x='index', y='singular values')) + geom_point())


def Q15():
    """

    :return:
    """
    X, y = load_data('kc_house_data.csv')
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    plot_singular_values(s)


def Q16():
    """

    :return:
    """
    X, y = load_data('kc_house_data.csv')
    # randomize idx:
    test_idx_array = random.sample(range(0, 99), 25)
    train_idx_array = np.setdiff1d(np.arange(100), test_idx_array)

    # get test and train data sets- X,y:
    # test_data = X[test_idx_array]
    # test_response = y[test_idx_array]

    train_data = X[train_idx_array]
    train_response = y[train_idx_array]

    errors = []
    ps = np.arange(1, 101)

    for p in ps:
        num_of_samples = int(round(p/100 * 75))
        X = train_data[:num_of_samples]
        y = train_response[:num_of_samples].reshape(num_of_samples, 1)

        w_hat, s = fit_linear_regression(X.T, y)
        y_hat = predict(X, w_hat)
        errors.append(mse(y, y_hat))

    df = pd.DataFrame({'p%': ps, 'MSE': errors})
    print(ggplot(df, aes(x='p%', y='MSE')) + geom_point() +
          ggtitle("MSE over percentage of data taken from training set"))



def pearson(x, y):
    s_x = np.std(x)
    s_y = np.std(y)
    cov_x_y = np.sum(np.multiply( x - (np.sum(x) / len(x)), y - (np.sum(y) / len(y)))) / len(x)
    return cov_x_y / (s_x * s_y)


def feature_evaluation(X, y):
    for col in X.columns:
        if 'zipcode' not in col:
