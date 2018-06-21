import math
import numpy as np
import sys

# preprocessing
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import word2vec

# algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# evaluation
from sklearn.model_selection import KFold

# output
import matplotlib.pyplot as plt
import csv

# BASIC_ENCODING = "UTF-16"
BASIC_ENCODING = "UTF-8"

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
        kf = KFold(n_splits=k_fold)
        cr = ClassifierResults(k_fold, feature_extractor, algo)
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

            cr.train_answers[fold_idx] = train_answers
            cr.test_answers[fold_idx] = test_answers
            cr.train_results[fold_idx] = train_results
            cr.test_results[fold_idx] = test_results

            # if we don't need logs about accuracy, we may put the code that evaluates accuracy to the ClassifierResults
            cr.train_accuracy[fold_idx] = 100 * train_correct / len(train_answers)
            cr.test_accuracy[fold_idx] = 100 * test_correct / len(test_answers)

            print("Fold {}".format(fold_idx))
            print("Train avg accuracy = {:.3f}".format(cr.train_accuracy[fold_idx]))
            print("Test avg accuracy = {:.3f}".format(cr.test_accuracy[fold_idx]))
            fold_idx += 1

        return cr

    def output_results(self, classifier_results):
        fig, ax = plt.subplots()
        ax.boxplot([cr.test_accuracy for cr in classifier_results])
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

class ClassifierResults():
    def __init__(self, k_fold, feature_extractor, algo):
        self.k_fold = k_fold
        self.train_answers = np.empty(k_fold, dtype=object)
        self.test_answers = np.empty(k_fold, dtype=object)

        self.train_results = np.empty(k_fold, dtype=object)
        self.test_results = np.empty(k_fold, dtype=object)

        self.train_accuracy = np.empty(k_fold, dtype=float)
        self.test_accuracy = np.empty(k_fold, dtype=float)

        self.feature_extractor = feature_extractor.get_information()
        self.algo = algo.get_information()

    def save(self, file_name):
        writer = pd.ExcelWriter(file_name)
        tr = pd.DataFrame({"true_answers": self.train_answers, "results": self.train_results})
        te = pd.DataFrame({"true_answers": self.test_answers, "results": self.test_results})
        acc = pd.DataFrame({"train_accuracy": self.train_accuracy, "test_accuracy": self.test_accuracy})
        feature_info = pd.DataFrame(self.feature_extractor)
        algo_info = pd.DataFrame(self.algo)

        tr.to_excel(writer, "train_answers")
        te.to_excel(writer, "test_answers")
        acc.to_excel(writer, "accuracy")
        feature_info.to_excel(writer, "features_info")
        algo_info.to_excel(writer, "algorithm_info")

class DescriptionPreprocessor():
    @staticmethod
    def process_string(line, to_remove):
        table = str.maketrans(''.join(to_remove), ' ' * len(to_remove))
        line = line.translate(table)
        return " ".join(line.split())

    @staticmethod
    def simple_processing(line):
        to_remove = [':', ';', '?', '!', '(', ')', '[', ']']
        return DescriptionPreprocessor.process_string(line, to_remove)

    @staticmethod
    def special_processing(line):
        to_remove = [' '] # '.', '-', '/', '\\'
        line = DescriptionPreprocessor.simple_processing(line)
        return DescriptionPreprocessor.process_string(line, to_remove)

class NGramFeatureExtractor():
    def __init__(self, analyzer, ngram_range, max_features, preprocessor=None):
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.max_features = max_features

        self.vectorizer = CountVectorizer(analyzer=analyzer, tokenizer=None, ngram_range=ngram_range, max_features=max_features,
            preprocessor=preprocessor, encoding=BASIC_ENCODING)

    def extract(self, train_data, test_data):
        train_data_features = self.vectorizer.fit_transform(train_data)
        np.asarray(train_data_features)
        test_data_features = self.vectorizer.transform(test_data)
        np.asarray(test_data_features)
        return train_data_features, test_data_features

    def get_information(self):
        return {"name": "N-gram", "parameters": "type: {};\trange: {}\tfeatures: {}".format(self.analyzer, self.ngram_range, self.max_features)}

class Word2VecFeatureExtractor():
    def inialize(self, descriptions, num_features, workers, min_word_count, context, downsampling):
        self.num_features = num_features
        desc_words = np.empty(len(descriptions), dtype=object)
        for i in range(len(descriptions)):
            desc_words[i] = descriptions[i].split()

        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
            level=logging.INFO)

        self.model = word2vec.Word2Vec(desc_words, workers = workers, size = num_features,
            min_count = min_word_count, window = context, sample = downsampling)
        self.model.save("word2vec_models/features={}_mincount={}_context={}".format(num_features, min_word_count, context))
        return self

    def load(self, file_name):
        self.model = word2vec.Word2Vec.load(file_name)
        self.num_features = self.model.wv.syn0.shape[1]
        # print(self.num_features)
        return self

    def extract(self, train_data, test_data):
        return self.make_features(train_data), self.make_features(test_data)

    def make_features(self, descriptions):
        desc_words = np.empty(len(descriptions), dtype=object)
        for i in range(len(descriptions)):
            desc_words[i] = descriptions[i].split()

        feature_vectors = np.empty((len(desc_words), self.num_features), dtype=object)
        for i in range(len(desc_words)):
            feature_vectors[i] = self.make_feature(desc_words[i])

        # print(feature_vectors.shape)
        # quit()
        return feature_vectors

    def make_feature(self, words):
        res = np.zeros((self.num_features,), dtype="float32")
        index2word_set = set(self.model.wv.index2word)

        nwords = 0
        for word in words:
            if word in index2word_set:
                nwords = nwords + 1
                res = np.add(res, self.model[word])

        if (nwords != 0):
            res = np.divide(res, nwords)

        # print(res.shape)
        # quit()
        return res

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
        self.model = GaussianNB()

    def fit(self, train_features, train_answers):
        # print(train_features.shape)
        # print(train_answers.shape)
        self.model = self.model.fit(train_features.todense(), train_answers)

    def predict(self, test_features):
        return self.model.predict(test_features.todense())

    def get_information(self):
        return {"name": "Naive Bayes", "parameters": ""}

class KNeighborsAlgorithm():
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
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

    def get_information(self):
        return {"name": "K Neighbors", "parameters": "neighbors: {}".format(self.n_neighbors)}

class LogRegressionAlgorithm():
    def __init__(self, penalty='l1', tol=0.01):
        self.penalty = penalty
        self.tol = tol
        self.model = LogisticRegression(penalty=penalty, tol=tol)

    def fit(self, train_features, train_answers):
        # print(train_features.shape)
        # print(train_answers.shape)
        self.model = self.model.fit(train_features, train_answers)

    def predict(self, test_features):
        return self.model.predict(test_features)

    def get_information(self):
        return {"name": "Logistic Regression", "parameters": "penalty: {};tolerance: ".format(self.n_neighbors, self.tol)}

def main():
    ctg = Categorizer()
    ctg.read("data/utf-8/2_final_tables/.csv", "data/categories_1000.csv", cat_cnt = 10)

    # word2vec_extractor = Word2VecFeatureExtractor().initialize(ctg.descriptions, 300, 4, 3, 2, 1e-3)
    word2vec_extractor = Word2VecFeatureExtractor().load("word2vec_models/features=300_mincount=3_context=2")

    unigram_word = NGramFeatureExtractor("word", (1, 1), 3000, DescriptionPreprocessor.simple_processing)
    bigram_word = NGramFeatureExtractor("word", (2, 2), 3000, DescriptionPreprocessor.simple_processing)
    bigram_char = NGramFeatureExtractor("char", (2, 3), 3000, DescriptionPreprocessor.special_processing)
    bigram_char_spaces = NGramFeatureExtractor("char", (2, 3), 3000, DescriptionPreprocessor.simple_processing)

    cr_forest = ctg.cross_validate(4, unigram_word, RandomForestAlgorithm(10))
    cr_forest.save('forest.xlsx')
    quit()

    features = [unigram_word, bigram_word, bigram_char, bigram_char_spaces]
    algorithms = [RandomForestAlgorithm(10), LogRegressionAlgorithm()]

    cv_results = np.empty((len(features), len(algorithms)), dtype=object)

    for f in range(len(features)):
        for a in range(len(algorithms)):
            cv_results[f][a] = ctg.cross_validate(4, features[f], algorithms[a])

    ctg.output_results([row[0] for row in cv_results])

    # quit()
    # bigram.extract()
    # quit()

    # cr_forest = ctg.cross_validate(4, word2vec_extractor, RandomForestAlgorithm(10))
    # cr_regr = ctg.cross_validate(4, word2vec_extractor, LogRegressionAlgorithm())

    # ctg.cross_validate(4, NGramFeatureExtractor("word", (1, 1), 3000), RandomForestAlgorithm(10))
    # ctg.cross_validate(4, NGramFeatureExtractor("word", (1, 1), 1000), NaiveBayesAlgorithm())
    # ctg.cross_validate(4, NGramFeatureExtractor("word", (1, 1), 1000), KNeighborsAlgorithm(3))

    # cr_forest = ctg.cross_validate(4, unigram_word, RandomForestAlgorithm(10))
    # cr_regr = ctg.cross_validate(4, unigram_word, LogRegressionAlgorithm())

    # cr_forest = ctg.cross_validate(4, bigram_char, RandomForestAlgorithm(10))
    # cr_regr = ctg.cross_validate(4, bigram_char, LogRegressionAlgorithm())

    # ctg.cross_validate(4, NGramFeatureExtractor("char", (2, 2), 3000), RandomForestAlgorithm(30))

    # TODO: show 10 worst categories using box-plot

main()
