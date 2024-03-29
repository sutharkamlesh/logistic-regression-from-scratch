import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def train_test_split(X, y, test_size=0.3):
    m = X.shape[0]
    sample_size = int(np.ma.floor(X.shape[0] * test_size))
    indexes = np.arange(m)
    np.random.shuffle(indexes)
    return X[indexes[sample_size:]], X[indexes[:sample_size]], y[indexes[sample_size:]], y[indexes[:sample_size]]


class CustomLogisticRegression:
    # TODO: Make this class work for multi-class problem
    def __init__(self, C=0.5, learning_rate=0.01, max_iter=100, penalty=None, sample_size=1):
        self.C = C
        self.max_iter = max_iter
        self.penalty = penalty
        self.learning_rate = learning_rate
        self.sample_size = sample_size
        self.weights = None
        self.bias = None
        self.history = {
            "costs": [],
            "accuracy": []
        }

    def predict_proba(self, X):
        return sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X, threshold=0.5):
        # TODO: Tune threshold to get maximum accuracy
        return self.predict_proba(X) > threshold

    @property
    def penalty_term(self):
        # TODO: Add l1 penalty
        if self.penalty == "l2":
            return self.C * self.weights
        elif self.penalty is None:
            return 0
        else:
            raise

    def sample(self, X, y):
        sample_idx = np.random.choice(X.shape[0], self.sample_size)
        return X[sample_idx], y[sample_idx]

    def update_weights(self, X, y):
        self.weights = self.weights - self.learning_rate * (np.dot(X.T, self.predict_proba(X) - y)) / X.shape[0]

    def update_bias(self, X, y):
        self.bias = self.bias - (self.learning_rate * np.mean(self.predict_proba(X) - y))

    def cost(self, y_true, y_pred):
        cost = np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        # TODO: Add Regulaization Term in cost function
        # if theta:
        #     cost += np.sum(np.abs(theta))
        return -1 * cost

    def score(self, y_true, y_predicted):
        if len(y_true) != len(y_predicted):
            raise
        elif type(y_true) == type(y_predicted) == np.ndarray:
            return sum((y_true & y_predicted) | [not i for i in (y_true | y_predicted)]) / len(y_true)

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.random.randn(n)
        self.bias = 0
        for _ in range(self.max_iter):
            sample_X, sample_y = self.sample(X, y)
            self.update_weights(sample_X, sample_y)
            self.update_bias(sample_X, sample_y)
            self.history.get("costs").append(self.cost(y, self.predict_proba(X)))
            self.history.get("accuracy").append(self.score(y, self.predict(X)))


iris_data = datasets.load_breast_cancer()
X = iris_data.get("data")
m, n = X.shape
X = (X - X.mean(axis=0)) / X.std(axis=0)
# X = np.c_[np.ones(m), X]
y = (iris_data.get("target") == 0)

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = CustomLogisticRegression(sample_size=120, max_iter=1000)
clf.fit(X_train, y_train)
print("Weights: ", clf.weights, "\n", "Bias: ", clf.bias)

preds = clf.predict(X_test)
print("Accuracy: ", clf.score(y_test, preds))

print("Event rate:", sum(y_test) / len(y_test))

# Comparing the output of Scikit Learn Method
clf = LogisticRegression()
clf.fit(X_train, y_train)
print("Weights: ", clf.coef_, "\n", "Bias: ", clf.intercept_)

preds = clf.predict(X_test)
print("Accuracy: ", clf.score(y_test, preds))
