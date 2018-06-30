from shared import *
from sklearn.model_selection import KFold
from classifier_results import *

class Classifier():
    categories_to_print = set([
        # "938D00155D03330D11E7945E3C3A3C2D", # "ЗИП (для ценообразования)"
        # "B78C00155D03330711E647D81D5EE2A7", # "Комплектующая для сборки (СБОРКА)"
        # "B24E00155D030B1F11E270E6F5145C40", # "Наушники накладные"
        # "B24E00155D030B1F11E271C0D0DB7C45", # "Наушники вкладыши"
    ])
    categories_to_skip = set([\
        "938D00155D03330D11E7945E3C3A3C2D", # "ЗИП (для ценообразования)"
        "B78C00155D03330711E647D81D5EE2A7", # "Комплектующая для сборки (СБОРКА)"
        "89CB00155D03120211E47FCF329B18E8", # "Обувь"
        "90BF00155D03120211E47ABC9E8CFEC9", # "Сим-карты; Data-комплекты (сим-карта + 3/4G модем)"
        "B70B00155D03361B11E4B70570C9AC70", # "Коммутационное оборудование (Партнер)"
        "858400155D03361B11E4819DFCD38E2D", # "Автомобильная шина"
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
                if scipy.sparse.issparse(feature_vector):
                    feature_vector = feature_vector.todense()
                else:
                    feature_vector = [feature_vector]
                found_cat_idx = predictor.predict(feature_vector)[0]

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
        opp_cat_data.to_csv("output/opponents/1kk_real_type=0_catcnt={}.csv".format(self.cat_cnt))

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
            # break

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

