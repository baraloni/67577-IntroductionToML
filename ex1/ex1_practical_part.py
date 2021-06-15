import gaussian_3d as g3d
import numpy as np
import matplotlib.pyplot as plt
import math

epsilons = [0.5 ,0.25, 0.1, 0.01, 0.001] 
data = np.random.binomial(1, 0.25, (100000,1000))


def question11():
    """
    Uses the identity matrix as the covariance matrix to 
    generates random 3d points and plots them.
    :retunes: a np array of 5,000 3d points of shape (3, 50000)
    """
    x_y_z = g3d.x_y_z
    g3d.plot_3d(x_y_z)
    return x_y_z

def question12(x_y_z):
    """
    :param: x_y_z: array of 50,000 3d points
    Transforms the data from question11 (x_y_z) with the a scaling 
    matrix s, and plots the new points.
    :retunes: a np array of 5,000 3d points of shape (3, 50000)

    """
    s = np.matrix([[0.1, 0 , 0],
                  [0, 0.5, 0],
                  [0, 0, 2]])
    x_y_z = s * x_y_z
    g3d.plot_3d(x_y_z)
    return x_y_z

def question13(x_y_z):
    """
    :param: x_y_z: array of 50,000 3d points
    Multiplys the scaled data returned by question12 (x_y_z) by a random 
    orthogonal matrix, and plots the new points.
    :retunes: a np array of 5,000 3d points of shape (3, 50000)
    """
    q = g3d.get_orthogonal_matrix(3)
    x_y_z = np.dot(q, x_y_z)
    g3d.plot_3d(x_y_z)
    return x_y_z 

def question14(x_y_z):
    """
    :param: x_y_z: array of 50,000 3d points
    Plots the projection of the data returned from question14 (x_y_z)
    to the x, y axes.
    """
    g3d.plot_2d(np.array(x_y_z[:-1]))
    
def question15(x_y_z):
    """
    :param: x_y_z: array of 50,000 3d points
    Plots the projection of the x_y_z points which holds that -0.4 < z < 0.1,
    to the x, y axes. 
    """
    z = x_y_z[-1]
    g3d.plot_2d(np.array(x_y_z[:2, np.where((-0.4 < z) & (z < 0.1))]))


def question16a():
    fig = plt.figure()
        
    ms = np.arange(1, 1001)
    plt.xlabel("Ms")

    m_vals_matrix = np.cumsum(data[:5], axis=1) / ms
    plt.ylabel("Estimates")

    for m_val in m_vals_matrix:
        plt.plot(ms, m_val)
        
    plt.legend(("exp 1", "exp 2", "exp 3", "exp 4", "exp 5"), loc='upper right')
    plt.title("The estimation of coin tosses as a function of M (first 5 experiments)")
    plt.show()
    
def question16b():
    ms = np.arange(1, 1001)

    for e in epsilons:
        fig = plt.figure()
        
        plt.xlabel("Ms")

        c_bounds = np.clip(1 / (4 * ms * e**2), 0, 1)
        h_bounds = np.clip(2 / np.array([math.exp(2 * m * e**2) for m in ms]), 0, 1)
        plt.ylabel("Upper bounds")

        plt.plot(ms, c_bounds)
        plt.plot(ms, h_bounds)

        plt.legend(("Chebyshev Bound", "Hoeﬀding Bound"), loc='upper right')
        plt.title("The upper bound as a function of M, with epsilon=" + str(e))
        plt.show()
        
    
def question16c():
        
    ms = np.arange(1, 1001)
    m_vals_matrix = np.abs(np.cumsum(data, axis=1) / ms - 0.25)
    
    for e in epsilons:
        fig = plt.figure()
        
        plt.xlabel("Ms")
            
        c_bounds = np.clip(1 / (4 * ms * e**2), 0, 1)
        h_bounds = np.clip(2 / np.array([math.exp(2 * m * e**2) for m in ms]), 0, 1)

        binary_failures = np.where(m_vals_matrix >= e, 1, 0)
        fail_in_perc = np.clip(np.sum(binary_failures, axis=0) / 100000, 0, 1)
        plt.ylabel("Failure (precentage)")

        plt.plot(ms, c_bounds)
        plt.plot(ms, h_bounds)
        plt.plot(ms, fail_in_perc)

        plt.legend(("Chebyshev Bound", "Hoeﬀding Bound", "Failure"), loc='upper right')
        plt.title("The upper bound and the failure (percentage) as a function of M, with epsilon=" + str(e))
        plt.show()
       


if __name__ == "__main__":
    x_y_z = question11()
    
    x_y_z = question12(x_y_z)
    cov = np.cov(x_y_z)
#     print("cov matrix after scaling:\n", cov)
    
    x_y_z = question13(x_y_z)
    cov = np.cov(x_y_z)
#     print("cov matrix after scaling and rotating:\n", cov)
    
    question14(x_y_z)
    
    question15(x_y_z)
    
    question16a()
    
    question16b()

    question16c()

