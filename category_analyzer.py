import math
import numpy as np
import sys
import matplotlib.pyplot as plt

import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

BASIC_ENCODING = "UTF-16"

class Categorizer():
    categories_to_output = set([
        # "9DC3001E6728A5A411E5A84BD6487A11",
    ])
    categories_to_skip = set([\
        "AB470002B3552D7511DABBA37AA42F4A", # different goods without any visible connection between them
        "AB470002B3552D7511DABBA4A64D4192",
    ])

    def read(self, data_file, categories_file, cat_cnt):
        print("Reading data from files '{}' and '{}'".format(data_file, categories_file))

        # chunksize = 10 ** 6
        # for products_data in pd.read_csv(data_file, encoding=BASIC_ENCODING, chunksize=chunksize):
        #     continue

        # quit()

        products_data = pd.read_csv(data_file, encoding=BASIC_ENCODING)
        categories_data = pd.read_csv(categories_file, sep="\t", encoding=BASIC_ENCODING)
        categories = np.array(categories_data.loc[:, "_ParentIDRRef"], dtype=str)

        self.cat_cnt = cat_cnt
        self.cat_names = np.empty(cat_cnt, dtype=object)

        print("Preparing train and test data")
        self.descriptions = np.array([], dtype=str)
        self.categories_idx = np.array([], dtype=int)

        cat_idx, category_idx = -1, -1
        while (cat_idx + 1 < cat_cnt and category_idx + 1 < len(categories_data)):
            category_idx += 1
            if (categories[category_idx] in self.categories_to_skip):
                continue

            cat_idx += 1
            self.cat_names[cat_idx] = categories[category_idx]
            cat_name = self.cat_names[cat_idx]

            cur_category_descriptions = np.array(products_data[products_data["_ParentIDRRef"] == cat_name].loc[:, "_Description"])
            sz = cur_category_descriptions.shape[0]

            self.descriptions = np.append(self.descriptions, cur_category_descriptions)
            self.categories_idx = np.append(self.categories_idx, np.full(sz, cat_idx, dtype=int))

            if (cat_name in self.categories_to_output):
                with open("categories_descriptions/{}.txt".format(cat_name), "w", encoding="UTF-16") as f:
                    f.writelines(["{}\n".format(item) for item in cur])
                # quit()

        # shuffle and lower descriptions
        perm = np.random.permutation(len(self.descriptions))
        self.descriptions = np.array(list(map(lambda x: str(x).lower(), self.descriptions[perm])))
        self.categories_idx = self.categories_idx[perm]
        return self

    def cross_validate(self, k_fold, feature_extractor, algo):
        # print(self.descriptions.shape)
        self.k_fold = k_fold
        kf = KFold(n_splits=k_fold)
        fold_idx = 0
        self.train_answers = np.empty(k_fold, dtype=object)
        self.test_answers = np.empty(k_fold, dtype=object)

        self.train_results = np.empty(k_fold, dtype=object)
        self.test_results = np.empty(k_fold, dtype=object)

        self.train_accuracy = np.empty(k_fold, dtype=float)
        self.test_accuracy = np.empty(k_fold, dtype=float)

        for train_idx, test_idx in kf.split(self.descriptions):
            train_data, train_answers = self.descriptions[train_idx], self.categories_idx[train_idx]
            test_data, test_answers = self.descriptions[test_idx], self.categories_idx[test_idx]

            train_features, test_features = feature_extractor.extract(train_data, test_data)
            algo.fit(train_features, train_answers)

            train_results = algo.predict(train_features)
            test_results = algo.predict(test_features)

            train_correct = len([i for i, j in zip(train_answers, train_results) if i == j])
            test_correct = len([i for i, j in zip(test_answers, test_results) if i == j])

            self.train_answers[fold_idx] = train_answers
            self.test_answers[fold_idx] = test_answers
            self.train_results[fold_idx] = train_results
            self.test_results[fold_idx] = test_results

            self.train_accuracy[fold_idx] = 100 * train_correct / len(train_answers)
            self.test_accuracy[fold_idx] = 100 * test_correct / len(test_answers)

            print("Fold {}".format(fold_idx))
            print("Train avg accuracy = {:.3f}".format(self.train_accuracy[fold_idx]))
            print("Test avg accuracy = {:.3f}".format(self.test_accuracy[fold_idx]))
            fold_idx += 1

        return self

    def output_results(self):
        fig, ax = plt.subplots()
        ax.boxplot([self.train_accuracy, self.test_accuracy])
        plt.show()
        quit()

        print("\n{} most common categories".format(cat_cnt))

        total_correct = 0
        total_amount = len(test_answers)
        cat_df = pd.DataFrame(index=[i for i in range(cat_cnt)], columns=["name", "correct", "amount", "avg_correctness"])

        for i in range(cat_cnt):
            correct = 0
            amount = cat_test_range[i][1]

            for j in range(cat_test_range[i][0], cat_test_range[i][0] + amount):
                if (test_answers[j] == test_results[j]):
                    correct += 1

            cat_df.loc[i] = [cat_names[i], correct, amount, 100 * correct / amount];
            total_correct += correct
            print("Category {:03d} ({}) : Correct {} from {} = {:.3f} %".format(i, cat_names[i], correct, amount, 100 * correct / amount))

        print("\nCorrect {} from {} = {:.3f} %".format(total_correct, total_amount, 100 * total_correct / total_amount))

        print("Categories in order of ascending correctness\n")
        cat_df = cat_df.sort_values(by=["avg_correctness"], ascending=True)

        for i, name, correct, amount, avg_corr in cat_df.itertuples(index=True, name='Pandas'):
            print("Category {:03d} ({}) : Correct {} from {} = {:.3f} %".format(
                i, name, correct, amount, avg_corr))

class NGramFeatureExtractor():
    def __init__(self, analyzer, ngram_range, max_features, preprocessor=None):
        self.vectorizer = CountVectorizer(analyzer=analyzer, tokenizer=None, ngram_range=ngram_range, max_features=max_features,
            preprocessor=preprocessor, encoding=BASIC_ENCODING)

    def extract(self, train_data, test_data):
        train_data_features = self.vectorizer.fit_transform(train_data)
        np.asarray(train_data_features)
        test_data_features = self.vectorizer.transform(test_data)
        np.asarray(test_data_features)
        return train_data_features, test_data_features

class RandomForestAlgorithm():
    def __init__(self, n_estimators, n_jobs=1, verbose=0):
        self.model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs, verbose=verbose)

    def fit(self, train_features, train_answers):
        self.model.fit(train_features, train_answers)

    def predict(self, test_features):
        return self.model.predict(test_features)

class NaiveBayesAlgorithm():
    def __init__(self):
        self.model = GaussianNB()

    def fit(self, train_features, train_answers):
        # print(train_features.shape)
        # print(train_answers.shape)
        self.model = self.model.fit(train_features.todense(), train_answers)

    def predict(self, test_features):
        return self.model.predict(test_features.todense())

class KNeighborsAlgorithm():
    def __init__(self, n_neighbors):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, train_features, train_answers):
        # print(train_features.shape)
        # print(train_answers.shape)
        self.model.fit(train_features, train_answers)
        print("Fit done")

    def predict(self, test_features):
        # print(type(test_features))
        # print(test_features.shape)
        # print(test_features.getnnz(), test_features[0].getnnz())
        # print(test_features.size)
        # print(test_features.dtype.itemsize)
        # print(test_features.size * test_features.dtype.itemsize)
        # tmp = test_features
        # print(sys.getsizeof(tmp))
        # print(tmp.shape)
        # tmp = test_features.todense()
        # print(len(tmp) * len(tmp[0]) * sys.getsizeof(tmp[0][0]))
        # print(tmp.shape)
        return self.model.predict(test_features)

class LogRegressionAlgorithm():
    def __init__(self, penalty='l1', tol=0.01):
        self.model = LogisticRegression(penalty=penalty, tol=tol)

    def fit(self, train_features, train_answers):
        # print(train_features.shape)
        # print(train_answers.shape)
        self.model = self.model.fit(train_features, train_answers)

    def predict(self, test_features):
        return self.model.predict(test_features)

def main():
    ctg = Categorizer()

    ctg.read("data/1_valid.csv", "data/categories_1000.csv", cat_cnt = 100)

    # print("done")
    # ctg.cross_validate(4, NGramFeatureExtractor("word", (1, 1), 3000), RandomForestAlgorithm(10))
    # ctg.cross_validate(4, NGramFeatureExtractor("word", (1, 1), 1000), NaiveBayesAlgorithm())
    # ctg.cross_validate(4, NGramFeatureExtractor("word", (1, 1), 1000), KNeighborsAlgorithm(3))
    ctg.cross_validate(4, NGramFeatureExtractor("word", (1, 1), 3000), LogRegressionAlgorithm())

    # ctg.cross_validate(4, NGramFeatureExtractor("char", (2, 2), 3000), RandomForestAlgorithm(30))
    ctg.output_results()

    # TODO: show 10 worst categories using box-plot

main()
