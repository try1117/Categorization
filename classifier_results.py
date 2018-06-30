from shared import *

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

    def save(self, file_name):
        writer = pd.ExcelWriter(file_name)
        total_acc = pd.DataFrame({"train_accuracy": 100 * self.total_train_score / self.total_train_size,
            "test_accuracy": 100 * self.total_test_score / self.total_test_size}, columns=["train_accuracy", "test_accuracy"])

        tr_sizes = np.array([v["size"] for v in self.train_sizes], dtype=int)
        te_sizes = np.array([v["size"] for v in self.test_sizes], dtype=int)

        # train_categories_acc = np.array([100 * self.categories_train_score[k] / tr_sizes for k in range(self.rounds)])
        # test_categories_acc = np.array([100 * self.categories_test_score[k] / te_sizes for k in range(self.rounds)])

        # print(self.categories_train_score)

        train_categories_acc = np.array(100 * self.categories_train_score[0] / tr_sizes)
        test_categories_acc = np.array(100 * self.categories_test_score[0] / te_sizes)
        train_test_delta = np.abs(train_categories_acc - test_categories_acc)

        # print(self.categories_train_score[0])

        data = list(zip(self.categories_train_score[0], tr_sizes, train_categories_acc,
            self.categories_test_score[0], te_sizes, test_categories_acc, train_test_delta))

        # print(data[0])
        # print(data)

        acc = pd.DataFrame(data, index=self.categories_name,
            columns=["Train correct", "Train size", "Train score", "Test correct", "Test size", "Test score", "Abs difference"])

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

        acc.to_excel(writer, "accuracy")
        tr_twins.to_excel(writer, "train categories-twins")
        te_twins.to_excel(writer, "test categories-twins")

        total_acc.to_excel(writer, "overall accuracy")
        feature_algo.to_excel(writer, "features and algorithm")
        writer.save()
