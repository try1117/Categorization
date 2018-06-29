import math
import numpy as np
import scipy
import sys
import pickle

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
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# evaluation
from sklearn.model_selection import KFold

# output
import matplotlib.pyplot as plt
import csv
from sklearn.manifold import TSNE
import utils

# BASIC_ENCODING = "UTF-16"
BASIC_ENCODING = "UTF-8"

class Categorizer():
    categories_to_print = set([
        # "938D00155D03330D11E7945E3C3A3C2D", # "ЗИП (для ценообразования)"
        # "B78C00155D03330711E647D81D5EE2A7", # "Комплектующая для сборки (СБОРКА)"
        "B24E00155D030B1F11E270E6F5145C40", # "Наушники накладные"
        "B24E00155D030B1F11E271C0D0DB7C45", # "Наушники вкладыши"
    ])
    categories_to_skip = set([\
        "938D00155D03330D11E7945E3C3A3C2D", # "ЗИП (для ценообразования)"
        "B78C00155D03330711E647D81D5EE2A7", # "Комплектующая для сборки (СБОРКА)"
    ])

    def read(self, data_file, categories_file, cat_cnt):
        print("Reading data from files '{}' and '{}'".format(data_file, categories_file))
        products_raw_data = pd.read_csv(data_file, encoding=BASIC_ENCODING)
        categories_raw_data = pd.read_csv(categories_file, encoding=BASIC_ENCODING)
        # cat_cnt = min(cat_cnt, categories_raw_data.shape[0])
        self.categories_data = pd.DataFrame({"id": np.empty(cat_cnt, dtype=str), "size": np.empty(cat_cnt, dtype=str),
            "name": np.empty(cat_cnt, dtype=str)}, columns=["id", "size", "name"])

        print("Reading descriptions")
        products_index = 0
        self.cat_cnt = cat_cnt
        self.descriptions = np.empty(cat_cnt, dtype=object)

        for i in range(cat_cnt):
            while products_raw_data.iloc[products_index]["category_id"] in self.categories_to_skip:
                products_index += 1

            cur_category_id = products_raw_data.iloc[products_index]["category_id"]
            cur_category_name = categories_raw_data[categories_raw_data["category_id"] == cur_category_id].iloc[0]["category_name"]

            left = products_index
            while (products_raw_data.iloc[products_index]["category_id"] == cur_category_id):
                products_index += 1

            self.descriptions[i] = np.array(products_raw_data.loc[left:products_index-1, "description"])
            self.categories_data.iloc[i] = [cur_category_id, products_index - left, cur_category_name]

            if cur_category_id in self.categories_to_print:
                with open("categories_descriptions/{}.txt".format(cur_category_id), "w", encoding=BASIC_ENCODING) as f:
                    f.writelines(["{}\n".format(item) for item in self.descriptions[i]])
                # quit()

            if ((i + 1) % 10 == 0):
                print("Processing | categories: {:02d} from {:02d} | descriptions: {}".format(i + 1, cat_cnt, products_index))
            # print(cat_real_size[i], cur_category_name)

        # shuffle and lower descriptions
        self.total_descriptions_cnt = 0
        for i in range(cat_cnt):
            perm = np.random.permutation(len(self.descriptions[i]))
            self.descriptions[i] = np.array(list(map(lambda x: str(x).lower(), self.descriptions[i][perm])))
            self.total_descriptions_cnt += len(self.descriptions[i])
        return self

    def read_opponents(self, data_file):
        print("Reading data from file '{}'".format(data_file))
        self.opp_data = pd.read_csv(data_file, encoding=BASIC_ENCODING)

    def category_exists(self, category_id):
        if not hasattr(self, "categories_id_to_index"):
            self.categories_id_to_index = dict(zip(self.categories_data["id"], range(self.cat_cnt)))
        return category_id in self.categories_id_to_index

    def category_id_to_index(self, category_id):
        if self.category_exists(category_id):
            return self.categories_id_to_index[category_id]
        return NaN

    def test_opponents(self, feature_extractor, predictor):
        print("Testing opponents")
        total_attempts, good_attempts = 0, 0
        cat_total = np.zeros(self.cat_cnt, dtype=int)
        cat_good = np.zeros(self.cat_cnt, dtype=int)

        for index, row in self.opp_data.iterrows():
            if (index % 50000 == 0):
                print("Done {} %".format(100 * index / self.opp_data.shape[0]))
            if row["type"] == 0 and not pd.isnull(row["category_id"]) and self.category_exists(row["category_id"]):
                real_cat_index = self.category_id_to_index(row["category_id"])
                total_attempts += 1
                cat_total[real_cat_index] += 1

                feature_vector = feature_extractor.transform([row["opp_product_description"]])[0]
                found_cat_idx = predictor.predict([feature_vector])[0]
                if self.categories_data.iloc[found_cat_idx]["id"] == row["category_id"]:
                    cat_good[found_cat_idx] += 1
                    good_attempts += 1

        opp_cat_data = pd.DataFrame({"name": np.array(self.categories_data["name"]),
            "correct": cat_good, "total": cat_total,
            "accuracy": list(map(lambda p: 0 if p[1] == 0 else 100 * p[0] / p[1], zip(cat_good, cat_total)))},
            columns=["name", "correct", "total", "accuracy"])

        print("Total amount of records {}".format(self.opp_data.shape[0]))
        print("Records which categories we know {} from {} = {:.2f} %".format(total_attempts, self.opp_data.shape[0],
            100 * total_attempts / self.opp_data.shape[0]))
        print("Total success {} from {} records = {:.2f} %".format(good_attempts, total_attempts, 100 * good_attempts / total_attempts))
        print(opp_cat_data)
        opp_cat_data.to_csv("output/opponents/1kk_type=0_catcnt={}.csv".format(self.cat_cnt))

    def k_fold_cross_validate(self, k_fold, feature_extractor, algo):
        kf = KFold(n_splits=k_fold)
        cr = ClassifierResults(k_fold, self.categories_data, feature_extractor, algo)
        fold_idx = 0

        print("{}-fold cross-validation".format(k_fold))
        print("Feature extractor: {}".format(feature_extractor.get_information()))
        print("Algorithm: {}".format(algo.get_information()))
        print("Descriptions amount: {}".format(len(self.descriptions)))

        for test_idx, train_idx in kf.split(self.descriptions):
            train_data, train_answers = self.descriptions[train_idx], self.desc_to_cat_idx[train_idx]
            test_data, test_answers = self.descriptions[test_idx], self.desc_to_cat_idx[test_idx]

            train_features, test_features = feature_extractor.extract(train_data, test_data)
            algo.fit(train_features, train_answers)
            # print("Fit length: {}".format(len(train_features)))
            # print("Fit length: {}".format(train_features.shape))

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

    def cross_validate(self, rounds, partition, max_category_trainset, feature_extractor, algo, verbose=1):
        print("{}-round cross-validation".format(rounds))
        print("Partition: {} | Max category trainset size: {}".format(partition, max_category_trainset))
        print("Feature extractor: {}".format(feature_extractor.get_information()))
        print("Algorithm: {}".format(algo.get_information()))

        cr = ClassifierResults(rounds, self.categories_data, feature_extractor, algo)
        cr.total_train_size, cr.total_test_size = 0, 0
        cr.train_sizes = np.full(self.cat_cnt, 0, dtype=dict)
        cr.test_sizes = np.full(self.cat_cnt, 0, dtype=dict)

        for i in range(self.cat_cnt):
            sz = len(self.descriptions[i])
            tr_sz = min(int(sz * partition), max_category_trainset)
            te_sz = sz - tr_sz
            cr.train_sizes[i] = {"left": cr.total_train_size, "right": cr.total_train_size + tr_sz, "size": tr_sz}
            cr.test_sizes[i] = {"left": cr.total_test_size, "right": cr.total_test_size + te_sz, "size": te_sz}
            cr.total_train_size += tr_sz
            cr.total_test_size += te_sz

        print("Total trainset size: {}".format(cr.total_train_size))
        print("Total testset size: {}".format(cr.total_test_size))

        for r in range(rounds):
            train_data = np.empty(cr.total_train_size, dtype=object)
            train_answers = np.empty(cr.total_train_size, dtype=int)
            permutations = np.empty(self.cat_cnt, dtype=object)

            for i in range(self.cat_cnt):
                permutations[i] = np.random.permutation(len(self.descriptions[i]))

            for i in range(self.cat_cnt):
                tr = self.descriptions[i][permutations[i][:cr.train_sizes[i]["size"]]]
                left = cr.train_sizes[i]["left"]
                right = cr.train_sizes[i]["right"]
                train_data[left:right] = tr
                train_answers[left:right] = i

            if verbose >= 2:
                print("Fitting data")
            # !!! can shuffle records before fitting them into model
            train_features = feature_extractor.fit_transform(train_data)
            algo.fit(train_features, train_answers)

            # FOR OPPONENTS TESTING
            break

            if verbose >= 2:
                print("Testing algorithm on trainset")

            cr.total_test_score[r] = 0
            cr.total_train_score[r] = 0
            # test algorithm on the same trainset
            for i in range(self.cat_cnt):
                train_data = self.descriptions[i][permutations[i][:cr.train_sizes[i]["size"]]]
                train_features = feature_extractor.transform(train_data)
                train_results = algo.predict(train_features)
                cr.categories_train_score[r][i] = len([res for res in train_results if res == i])
                cr.total_train_score[r] += cr.categories_train_score[r][i]

                # analyze our errors
                if r == 0:
                    cr.categories_train_twins[i] = {}
                    for res in train_results:
                        if res != i:
                            cr.categories_train_twins[i][res] = cr.categories_train_twins[i].get(res, 0) + 1

                if verbose >= 2:
                    print("Done {:02d} from {:02d}".format(i + 1, self.cat_cnt))

            if verbose >= 2:
                print("Testing algorithm on testset")

            # test algorithm on the testset
            for i in range(self.cat_cnt):
                test_data = self.descriptions[i][permutations[i][cr.train_sizes[i]["size"]:]]
                test_features = feature_extractor.transform(test_data)
                test_results = algo.predict(test_features)
                cr.categories_test_score[r][i] = len([res for res in test_results if res == i])
                cr.total_test_score[r] += cr.categories_test_score[r][i]

                # analyze our errors
                if r == 0:
                    cr.categories_test_twins[i] = {}
                    for res in test_results:
                        if res != i:
                            cr.categories_test_twins[i][res] = cr.categories_test_twins[i].get(res, 0) + 1

                if verbose >= 2:
                    print("Done {:02d} from {:02d}".format(i + 1, self.cat_cnt))

            print("Round {}".format(r + 1))
            print("Train avg accuracy = {:.3f}".format(100 * cr.total_train_score[r] / cr.total_train_size))
            print("Test avg accuracy = {:.3f}".format(100 * cr.total_test_score[r] / cr.total_test_size))

        return cr, algo

    def output_results(self, classifier_results):
        fig, ax = plt.subplots()
        ax.boxplot([cr.test_accuracy for cr in classifier_results])
        plt.show()
        quit()

    def draw_categories(self, draw_cnt, feature_extractor):
        x = np.array([], dtype=object)
        products_name = np.array([], dtype=str)
        y = np.array([], dtype=int)
        draw_cnt[0] = min(draw_cnt[0], len(self.descriptions))
        for i in range(draw_cnt[0]):
            x = np.append(x, self.descriptions[i][:draw_cnt[1]])
            y = np.append(y, [i] * draw_cnt[1])
            products_name = np.append(products_name, self.descriptions[i][:draw_cnt[1]])

        tsne = TSNE(n_components=2, random_state=0)
        x = feature_extractor.fit_transform(x)
        if scipy.sparse.issparse(x):
            x = x.todense()
        x_2d = tsne.fit_transform(x)

        categories_name = np.array(self.categories_data["name"][:draw_cnt[0]])
        utils.LabeledScatterPlot(x_2d, y, products_name, range(draw_cnt[0]), categories_name)

class ClassifierResults():
    def __init__(self, rounds, categories_data, feature_extractor, algo):
        self.rounds = rounds
        self.categories_name = np.array(categories_data["name"])
        self.categories_size = np.array(categories_data["size"])

        self.cat_cnt = len(self.categories_name)
        self.categories_test_score = np.zeros((rounds, self.cat_cnt), dtype=int)
        self.categories_train_score = np.zeros((rounds, self.cat_cnt), dtype=int)

        self.total_test_score = np.zeros(rounds, dtype=int)
        self.total_train_score = np.zeros(rounds, dtype=int)
        self.total_train_size, self.total_test_size = 0, 0

        self.categories_train_twins = np.empty(self.cat_cnt, dtype=dict)
        self.categories_test_twins = np.empty(self.cat_cnt, dtype=dict)

        self.feature_extractor_info = feature_extractor.get_information()
        self.algo_info = algo.get_information()

    def old_save(self, file_name):
        writer = pd.ExcelWriter(file_name)
        total_acc = pd.DataFrame({"train_accuracy": self.train_accuracy, "test_accuracy": self.test_accuracy}, columns=["train_accuracy", "test_accuracy"])

        cat_cnt = len(self.categories_name)
        train_categories_correct = np.zeros((self.k_fold, cat_cnt), dtype=int)
        test_categories_correct = np.zeros((self.k_fold, cat_cnt), dtype=int)

        train_categories_size = np.zeros((self.k_fold, cat_cnt), dtype=int)
        test_categories_size = np.zeros((self.k_fold, cat_cnt), dtype=int)

        for k in range(self.k_fold):
            for i in range(len(self.train_answers[k])):
                cur_cat = self.train_answers[k][i]
                if cur_cat == self.train_results[k][i]:
                    train_categories_correct[k][cur_cat] += 1
                train_categories_size[k][cur_cat] += 1

        for k in range(self.k_fold):
            for i in range(len(self.test_answers[k])):
                cur_cat = self.test_answers[k][i]
                if cur_cat == self.test_results[k][i]:
                    test_categories_correct[k][cur_cat] += 1
                test_categories_size[k][cur_cat] += 1

        train_categories_data = np.transpose([100 * train_categories_correct[k] / train_categories_size[k] for k in range(self.k_fold)])
        test_categories_data = np.transpose([100 * test_categories_correct[k] / test_categories_size[k] for k in range(self.k_fold)])
        train_test_delta = np.abs(train_categories_data - test_categories_data)

        train_categories_data = np.array([np.append(train_categories_data[i], [np.average(train_categories_data[i])]) for i in range(cat_cnt)])
        test_categories_data = np.array([np.append(test_categories_data[i], [np.average(test_categories_data[i])]) for i in range(cat_cnt)])
        train_test_delta = np.array([np.append(train_test_delta[i], [np.average(train_test_delta[i])]) for i in range(cat_cnt)])

        tr_df = pd.DataFrame(train_categories_data, index=self.categories_name,
            columns=["Fold {}".format(k + 1) for k in range(self.k_fold)] + ["Folds avg"])

        te_df = pd.DataFrame(test_categories_data, index=self.categories_name,
            columns=["Fold {}".format(k + 1) for k in range(self.k_fold)] + ["Folds avg"])

        delta_df = pd.DataFrame(train_test_delta, index=self.categories_name,
            columns=["Fold {}".format(k + 1) for k in range(self.k_fold)] + ["Folds avg"])

        feature_algo = pd.DataFrame([self.feature_extractor_info, self.algo_info], index=["Feature extractor", "Algorithm"], columns=["name", "parameters"])

        tr_df.to_excel(writer, "train accuracy")
        te_df.to_excel(writer, "test accuracy")
        delta_df.to_excel(writer, "delta between train and test")
        total_acc.to_excel(writer, "overall accuracy")
        feature_algo.to_excel(writer, "features and algorithm")
        writer.save()

    def save(self, file_name):
        writer = pd.ExcelWriter(file_name)
        total_acc = pd.DataFrame({"train_accuracy": 100 * self.total_train_score / self.total_train_size,
            "test_accuracy": 100 * self.total_test_score / self.total_test_size}, columns=["train_accuracy", "test_accuracy"])

        tr_sizes = np.array([v["size"] for v in self.train_sizes], dtype=int)
        te_sizes = np.array([v["size"] for v in self.test_sizes], dtype=int)

        train_categories_data = np.transpose([100 * self.categories_train_score[k] / tr_sizes for k in range(self.rounds)])
        test_categories_data = np.transpose([100 * self.categories_test_score[k] / te_sizes for k in range(self.rounds)])
        train_test_delta = np.abs(train_categories_data - test_categories_data)

        train_categories_data = np.array([np.append(train_categories_data[i], [np.average(train_categories_data[i])]) for i in range(self.cat_cnt)])
        test_categories_data = np.array([np.append(test_categories_data[i], [np.average(test_categories_data[i])]) for i in range(self.cat_cnt)])
        train_test_delta = np.array([np.append(train_test_delta[i], [np.average(train_test_delta[i])]) for i in range(self.cat_cnt)])

        tr_acc = pd.DataFrame(train_categories_data, index=self.categories_name,
            columns=["Fold {}".format(k + 1) for k in range(self.rounds)] + ["Folds avg"])

        te_acc = pd.DataFrame(test_categories_data, index=self.categories_name,
            columns=["Fold {}".format(k + 1) for k in range(self.rounds)] + ["Folds avg"])

        delta_acc = pd.DataFrame(train_test_delta, index=self.categories_name,
            columns=["Fold {}".format(k + 1) for k in range(self.rounds)] + ["Folds avg"])

        # print(self.categories_train_twins)
        # print(self.categories_test_twins)

        # for each category show top-3 the most similar to it categories
        train_twins_data = np.empty((self.cat_cnt, 3), dtype=object)
        test_twins_data = np.empty((self.cat_cnt, 3), dtype=object)
        for i in range(self.cat_cnt):
            sorted_by_value = sorted(self.categories_train_twins[i].items(), key=lambda kv: kv[1], reverse=True)
            for j in range(min(len(sorted_by_value), 3)):
                train_twins_data[i][j] = ("{:.02f}".format(100 * sorted_by_value[j][1] / tr_sizes[i]),
                    self.categories_name[sorted_by_value[j][0]])

            sorted_by_value = sorted(self.categories_test_twins[i].items(), key=lambda kv: kv[1], reverse=True)
            for j in range(min(len(sorted_by_value), 3)):
                test_twins_data[i][j] = ("{:.02f}".format(100 * sorted_by_value[j][1] / te_sizes[i]),
                    self.categories_name[sorted_by_value[j][0]])

        tr_twins = pd.DataFrame(train_twins_data, index=self.categories_name, columns=["Top-1", "Top-2", "Top-3"])
        te_twins = pd.DataFrame(test_twins_data, index=self.categories_name, columns=["Top-1", "Top-2", "Top-3"])

        feature_algo = pd.DataFrame([self.feature_extractor_info, self.algo_info], index=["Feature extractor", "Algorithm"], columns=["name", "parameters"])

        tr_acc.to_excel(writer, "train accuracy")
        te_acc.to_excel(writer, "test accuracy")
        delta_acc.to_excel(writer, "delta between train and test")
        tr_twins.to_excel(writer, "train categories-twins")
        te_twins.to_excel(writer, "test categories-twins")

        total_acc.to_excel(writer, "overall accuracy")
        feature_algo.to_excel(writer, "features and algorithm")
        writer.save()

class DescriptionPreprocessor():
    @staticmethod
    def process_string(line, to_remove):
        table = str.maketrans(''.join(to_remove), ' ' * len(to_remove))
        line = line.translate(table)
        return " ".join(line.split())

    @staticmethod
    def simple_processing(line):
        to_remove = [':', ';', '\'', '\"', '/', '\\', '?', '!', '(', ')', '[', ']']
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

    def fit_transform(self, data):
        data_features = self.vectorizer.fit_transform(data)
        np.asarray(data_features)
        return data_features

    def transform(self, data):
        data_features = self.vectorizer.transform(data)
        np.asarray(data_features)
        return data_features

    def get_information(self):
        return {"name": "N-gram", "parameters": "type: {};\trange: {};\tfeatures: {}".format(self.analyzer, self.ngram_range, self.max_features)}

class Word2VecFeatureExtractor():
    def initialize(self, descriptions, total_descriptions_cnt, num_features, workers, min_word_count, context, downsampling):
        self.num_features = num_features
        self.min_word_count = min_word_count
        self.context = context

        desc_words = np.empty(total_descriptions_cnt, dtype=object)
        idx = 0
        for cat_desc in descriptions:
            for desc in cat_desc:
                desc_words[idx] = DescriptionPreprocessor.simple_processing(desc).split()
                idx += 1

        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
            level=logging.INFO)

        self.model = word2vec.Word2Vec(desc_words, workers = workers, size = num_features,
            min_count = min_word_count, window = context, sample = downsampling)
        self.model.save("word2vec_models/features={}_mincount={}_context={}".format(num_features, min_word_count, context))
        self.index2word_set = set(self.model.wv.index2word)
        return self

    def get_information(self):
        return {"name": "Word2Vec", "parameters": "features: {};\tmin_word_count: {};\tcontext: {}".format(self.num_features,\
            self.min_word_count, self.context)}

    def load(self, file_name):
        self.model = word2vec.Word2Vec.load(file_name)
        self.num_features = self.model.wv.syn0.shape[1]
        self.min_word_count = ''
        self.context = ''
        self.index2word_set = set(self.model.wv.index2word)
        # print(self.num_features)
        return self

    def extract(self, train_data, test_data):
        return self.make_features(train_data), self.make_features(test_data)

    def fit_transform(self, data):
        return self.make_features(data)

    def transform(self, data):
        return self.make_features(data)

    def make_features(self, descriptions):
        # print("Processing descriptions in word2vec")
        # print(descriptions.shape)
        desc_words = np.empty(len(descriptions), dtype=object)
        for i in range(len(descriptions)):
            desc_words[i] = DescriptionPreprocessor.simple_processing(descriptions[i]).split()

        # print(desc_words.shape)
        # quit()

        # print("Making feature vectors")
        # print(desc_words.shape)
        feature_vectors = np.empty((len(desc_words), self.num_features), dtype=object)
        for i in range(len(desc_words)):
            feature_vectors[i] = self.make_feature(desc_words[i])
            # small arrays won't trigger this message
            if i >= 1000 and i % 1000 == 0:
                print("Done {} from {}".format(i + 1, len(desc_words)))

        # print(feature_vectors.shape)
        # print("Done")
        return feature_vectors

    def make_feature(self, words):
        res = np.zeros((self.num_features,), dtype="float32")
        nwords = 0

        for word in words:
            if word in self.index2word_set:
                nwords = nwords + 1
                res = np.add(res, self.model[word])

        if (nwords != 0):
            res = np.divide(res, nwords)
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

def main():
    ctg = Categorizer()
    ctg.read("data/utf-8/2_final_tables/products_dns_short_sorted_utf8.csv", "data/utf-8/2_final_tables/categories_utf8.csv", cat_cnt = 200)
    ctg.read_opponents("data/utf-8/2_final_tables/mega_sub_1e6_filled_07.csv")
    # quit()

    # word2vec_extractor = Word2VecFeatureExtractor().initialize(ctg.descriptions, ctg.total_descriptions_cnt, 300, 4, 3, 2, 1e-3)
    word2vec_extractor = Word2VecFeatureExtractor().load("word2vec_models/features=300_mincount=3_context=2")
    # print(len(word2vec_extractor.index2word_set))
    # print(word2vec_extractor.index2word_set)
    # quit()

    unigram_word = NGramFeatureExtractor("word", (1, 1), 1000, DescriptionPreprocessor.simple_processing)
    bigram_word = NGramFeatureExtractor("word", (2, 2), 1000, DescriptionPreprocessor.simple_processing)
    biuni_word = NGramFeatureExtractor("word", (1, 2), 2000, DescriptionPreprocessor.simple_processing)
    bigram_char = NGramFeatureExtractor("char", (2, 3), 1000, DescriptionPreprocessor.special_processing)
    bigram_char_spaces = NGramFeatureExtractor("char", (2, 3), 3000, DescriptionPreprocessor.simple_processing)

    # ctg.draw_categories([20, 25], word2vec_extractor)
    # quit()

    # cr_bay = ctg.cross_validate(4, 0.75, 1000, word2vec_extractor, NaiveBayesAlgorithm(), 2)
    # cr_bay.save("output/bay_cat=30_1000_word2vec_features=300.xlsx")
    # quit()

    # cr_neigh = ctg.cross_validate(1, 0.75, 1000, bigram_char, KNeighborsAlgorithm(3), 2)
    # cr_neigh.save("output/neigh=3_cat=30_char_features=1000.xlsx")
    # quit()

    # cr_regr = ctg.cross_validate(4, 0.75, 1000, unigram_word, LogRegressionAlgorithm(0.001))
    # cr_regr.save("output/regr_tol=0.001_cat=100_word_features=1000.xlsx")
    # quit()

    cr_forest, predictor = ctg.cross_validate(1, 0.75, 1000, word2vec_extractor, RandomForestAlgorithm(10), 1)
    ctg.test_opponents(word2vec_extractor, predictor)
    quit()
    # cr_forest.save("output/forest_cat=40_word2vec_features=300.xlsx")
    # quit()

    # cr_forest = ctg.cross_validate(4, unigram_word, RandomForestAlgorithm(10))
    # cr_forest.save("output/forest.xlsx")
    # quit()

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
