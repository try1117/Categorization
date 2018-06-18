import math
import numpy as np

import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

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

    def cross_validate(self, k_fold, feature_extractor, algo):
        # print(self.descriptions.shape)
        kf = KFold(n_splits=k_fold)
        fold_idx = 0
        for train_idx, test_idx in kf.split(self.descriptions):
            train_data, train_answers = self.descriptions[train_idx], self.categories_idx[train_idx]
            test_data, test_answers = self.descriptions[test_idx], self.categories_idx[test_idx]

            train_features, test_features = feature_extractor.extract(train_data, test_data)
            algo.fit(train_features, train_answers)

            train_results = algo.predict(train_features)
            test_results = algo.predict(test_features)

            train_correct = len([i for i, j in zip(train_answers, train_results) if i == j])
            test_correct = len([i for i, j in zip(test_answers, test_results) if i == j])

            fold_idx += 1
            print("Fold {}".format(fold_idx))
            print("Train avg accuracy = {:.3f}".format(100 * train_correct / len(train_answers)))
            print("Test avg accuracy = {:.3f}".format(100 * test_correct / len(test_answers)))

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
        self.model = self.model.fit(train_features, train_answers)

    def predict(self, test_features):
        return self.model.predict(test_features)

def main():
    ctg = Categorizer()
    ctg.read("data/1_valid.csv", "data/categories_1000.csv", cat_cnt = 100)
    ctg.cross_validate(4, NGramFeatureExtractor("word", (1, 1), 3000), RandomForestAlgorithm(30))

main()
