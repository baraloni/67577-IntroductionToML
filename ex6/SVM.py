import numpy as np


def subgradient(x, y, w, b, gamma):
  m, d = x.shape
  gradients = [x[i] * y[i] for i in range(m) if y[i] * np.dot(x[i], w) < 1]
  gradients.append(np.array([0]*d)) # handles the case where the list is empty
  return gamma*w - np.sum(gradients, axis=0).reshape(d, 1)


def gd(data, label, iters, eta, w, gamma):
  m, d = data.shape
  # x = np.copy(data)
  # x = np.hstack((x, np.ones(m).reshape(m, 1)))
  w_t = w

  w_s = [w_t]

  for t in range(iters):
      gradient = np.append(subgradient(data, label, w_t[:-1], w_t[-1], gamma), [w[-1]]).reshape(d + 1, 1)
      w_t = w_t - eta * gradient
      w_s.append(w_t)
  return w_s


def sgd(data, label, iters, eta, w, batch, gamma):
  m, d = data.shape
  # x = np.copy(data)
  # x = np.hstack((x, np.ones(m).reshape(m, 1)))
  w_s = [w]

  w_t = w
  for t in range(iters):
      indices = np.random.choice(m, batch, replace=False)
      gradient = np.append(subgradient(data[indices], label[indices], w_t[:-1], w_t[-1], gamma), [w[-1]]).reshape(d + 1, 1)
      w_t -= eta * gradient
      w_s.append(w_t)
  return w_s

def test_error(w, test_data, test_labels):
  m, d = test_data.shape
  h_labels = np.sign(np.dot(test_data, w[:-1]) + w[-1])
  return len(np.where(h_labels != test_labels)) / m