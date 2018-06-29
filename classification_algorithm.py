from shared import *

# algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

class RandomForestAlgorithm():
    def __init__(self, n_estimators, n_jobs=1, verbose=0):
        self.n_estimators = n_estimators
        self.model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, verbose=verbose)

    def fit(self, train_features, train_answers):
        # print(train_features.shape)
        self.model.fit(train_features, train_answers)

    def predict(self, test_features):
        return self.model.predict(test_features)

    def get_information(self):
        return {"name": "Random Forest", "parameters": "estimators: {}".format(self.n_estimators)}

class NaiveBayesAlgorithm():
    def __init__(self):
        # self.model = GaussianNB()
        self.model = MultinomialNB()

    def fit(self, train_features, train_answers):
        # print(train_features.shape)
        # print(train_answers.shape)
        # self.model = self.model.fit(train_features.todense(), train_answers)
        self.model = self.model.fit(train_features, train_answers)

    def predict(self, test_features):
        # return self.model.predict(test_features.todense())
        return self.model.predict(test_features)

    def get_information(self):
        return {"name": "Naive Bayes", "parameters": ""}

class KNeighborsAlgorithm():
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, train_features, train_answers):
        self.model.fit(train_features, train_answers)
        print("Fit done")

    def predict(self, test_features):
        total_sz = test_features.shape[0]
        block_sz = 1000
        res = np.empty(total_sz, dtype=object)
        for i in range(total_sz // block_sz):
            L = block_sz * i
            R = min(total_sz, block_sz * (i + 1))
            res[L:R] = self.model.predict(test_features[L:R])

        return res
        # gives error: array is too big
        # return self.model.predict(test_features)

    def get_information(self):
        return {"name": "K Neighbors", "parameters": "neighbors: {}".format(self.n_neighbors)}

class LogRegressionAlgorithm():
    def __init__(self, tol=0.01, penalty='l1', ):
        self.tol = tol
        self.penalty = penalty
        self.model = LogisticRegression(penalty=penalty, tol=tol)

    def fit(self, train_features, train_answers):
        self.model = self.model.fit(train_features, train_answers)

    def predict(self, test_features):
        return self.model.predict(test_features)

    def get_information(self):
        return {"name": "Logistic Regression", "parameters": "penalty: {};tolerance: ".format(self.penalty, self.tol)}
