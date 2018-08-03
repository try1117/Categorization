from shared import *
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import word2vec
import gc

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
    def initialize(self, descriptions, total_descriptions_cnt, desc_in_categories=True, num_features=300, workers=1, min_word_count=10, context=2, downsampling=1e-3):
        self.num_features = num_features
        self.min_word_count = min_word_count
        self.context = context

        desc_words = np.empty(total_descriptions_cnt, dtype=object)
        if desc_in_categories:
            idx = 0
            for cat_desc in descriptions:
                for desc in cat_desc:
                    desc_words[idx] = DescriptionPreprocessor.simple_processing(desc).split()
                    idx += 1
        else:
            for i in range(len(descriptions)):
                desc_words[i] = DescriptionPreprocessor.simple_processing(str(descriptions[i])).split()

        import logging
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
            level=logging.INFO)

        print(workers, num_features, min_word_count, context, downsampling)
        print(desc_words)
        gc.collect()
        # with open("word2vec_models/all_descriptions.txt", "w") as f:
            # np.save(f, desc_words)
        # np.savetxt("word2vec_models/all_descriptions.csv", desc_words, delimiter=",", fmt="%s")
        # quit()

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
