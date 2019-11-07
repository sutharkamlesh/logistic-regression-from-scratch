import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split

# Initializig Dummy Datasets
# m = 1000    # Number of Intances
# n = 9       # Number of Variables
# X = np.random.randn(m, n)
# X = np.c_[np.ones(m), X]
# y = np.random.choice([0, 1], m, replace=True)


def train_test_split(X, y, test_size=0.3):
    m = X.shape[0]
    sample_size = int(np.ma.floor(X.shape[0]*test_size))
    indexes = np.arange(m)
    np.random.shuffle(indexes)
    return X[indexes[sample_size:]], X[indexes[:sample_size]], y[indexes[sample_size:]], y[indexes[:sample_size]]


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def predict(X, theta):
    return sigmoid(np.dot(X, theta))


def cost_function(y_true, y_pred):
    cost = 0
    m = len(y_true)
    for i in range(m):
        cost += y_true[i] * np.log(y_pred[i]) + (1 - y_true[i]) * np.log(1 - y_pred[i])
    # TODO: Add Regulaization Term in cost function
    # if theta:
    #     cost += np.sum(np.abs(theta))
    return -1 * cost / m


def gradient(theta, X, y):
    return (np.dot(X.T, predict(X, theta) - y) + theta) / X.shape[0]


# eta = 0.1
# num_iter = 100
# theta = np.random.randn(n+1)
# theta_history = np.zeros((num_iter, n+1))
# for k in range(num_iter):
#     theta = theta - eta * gradient(theta, X, y)
#     theta_history[k, :] = theta


iris_data = datasets.load_iris()
X = iris_data.get("data")
m, n = X.shape
X = np.c_[np.ones(m), X]
y = (iris_data.get("target") == 0)

eta = 0.1
tol = 1
num_iter = 10000
theta = np.random.randn(n + 1)
theta_history = np.zeros((num_iter, n + 1))
costs = []
gradients = []
tolerence = 1
k = 0
while tol > 0.0001 and k < num_iter:
    print(1)
    _, X_sample, _, y_sample = train_test_split(X, y, test_size=1)
    grad = gradient(theta, X_sample, y_sample)
    theta = theta - eta * grad
    costs.append(cost_function(y_true=y_sample, y_pred=predict(X_sample, theta)))
    gradients.append(grad)
    theta_history[k, :] = theta
    tol = costs[k]
    k += 1

plt.plot(range(num_iter), costs)

