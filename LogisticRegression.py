import numpy as np
from sklearn import datasets


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def train_test_split(X, y, test_size=0.3):
    m = X.shape[0]
    sample_size = int(np.ma.floor(X.shape[0] * test_size))
    indexes = np.arange(m)
    np.random.shuffle(indexes)
    return X[indexes[sample_size:]], X[indexes[:sample_size]], y[indexes[sample_size:]], y[indexes[:sample_size]]


class LogisticRegression():
    # TODO: Make this class work for multi-class problem
    def __init__(self, C = 0.5, learning_rate=0.01, max_iter=100, penalty=None):
        self.C = C
        self.max_iter = max_iter
        self.penalty = penalty
        self.learning_rate = learning_rate
        self.weights = None

    def predict_proba(self, X):
        return sigmoid(np.dot(X, self.weights))

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

    def update_weights(self, X, y):
        self.weights = self.weights - self.learning_rate * (np.dot(X.T, self.predict_proba(X) - y) + self.penalty_term) / \
                        X.shape[0]


    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.random.randn(n)
        for _ in range(self.max_iter):
            self.update_weights(X, y)

    def score(self, y_true, y_predicted):
        if len(y_true) != len(y_predicted):
            raise
        elif type(y_true) == type(y_predicted) == np.ndarray:
            return sum((y_true & y_predicted) | [ not i for i in (y_true | y_predicted)])/len(y_true)


iris_data = datasets.load_iris()
X = iris_data.get("data")
m, n = X.shape
X = (X - X.mean(axis=0))/X.std(axis=0)
X = np.c_[np.ones(m), X]
y = (iris_data.get("target") == 0)

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = LogisticRegression(penalty="l2")
clf.fit(X_train, y_train)
print("Weights: ", clf.weights)
preds = clf.predict(X_test)
print("Accuracy: ", clf.score(y_test, preds))