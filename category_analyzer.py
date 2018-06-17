import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
import math
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# TODO: more than just 'lower'
def clean_description(description):
    return str(description).lower()

def main():
    print("Reading data from .csv file")
    data = pd.read_csv("data/1_valid.csv", encoding="UTF-16")
    categories_full_data = pd.read_csv("data/categories_1000.csv", sep="\t", encoding="UTF-16")
    categories_names = np.array(categories_full_data.loc[:, "_ParentIDRRef"], dtype=str)

    cat_cnt = 100
    cat_names = np.empty(cat_cnt, dtype=object)
    cat_test_range = np.full((cat_cnt, 2), (0, 0), dtype=object) # (left index, size)

    train_data = np.array([], dtype=str)
    train_answers = np.array([], dtype=int)

    test_data = np.array([], dtype=str)
    test_answers = np.array([], dtype=int)

    training_part = 0.7
    print("Preparing train and test data")

    categories_to_output = set([
        # "81A800155D03361B11E5B9A7810E8D1E",
        # "9B10001517D7BDE111E42760C1FF05C5",
        # "AB470002B3552D7511DABBA4A64D4192",
        # "8D6700155D03361B11E48FD21C46F3CA",
        # "8F1D001517C5264411E4162B60C94D56",
        "AFBE00155D03330D11E75587CDA72B14",
    ])
    categories_to_skip = set([\
        "AB470002B3552D7511DABBA37AA42F4A", # different goods without any visible connection between them
        "AB470002B3552D7511DABBA4A64D4192",
    ])

    cat_idx, category_idx = -1, -1
    while (cat_idx + 1 < cat_cnt and category_idx + 1 < len(categories_names)):
        category_idx += 1
        if (categories_names[category_idx] in categories_to_skip):
            continue

        cat_idx += 1
        cat_names[cat_idx] = categories_names[category_idx]
        cat_name = cat_names[cat_idx]

        cur = np.array(data[data["_ParentIDRRef"] == cat_name].loc[:, "_Description"])
        sz = cur.shape[0]
        training_sz = int(sz * training_part)
        test_sz = sz - training_sz

        cat_test_range[cat_idx] = (len(test_data), test_sz)
        # shuffle array, as records in database are usually form blocks(of similar records)
        np.random.shuffle(cur)

        train_data = np.append(train_data, cur[:training_sz])
        train_answers = np.append(train_answers, np.full(training_sz, cat_idx, dtype=int))

        test_data = np.append(test_data, cur[training_sz:])
        test_answers = np.append(test_answers, np.full(test_sz, cat_idx, dtype=int))

        if (cat_name in categories_to_output):
            with open("categories_descriptions/{}.txt".format(cat_name), "w", encoding="UTF-16") as f:
                f.writelines(["{}\n".format(item) for item in cur])
            # quit()
    # quit()

    clean_train_data = np.array(list(map(clean_description, train_data)))
    clean_test_data = np.array(list(map(clean_description, test_data)))

    print("Extracting feactures from data")
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 3000)
    train_data_features = vectorizer.fit_transform(clean_train_data)
    np.asarray(train_data_features)
    test_data_features = vectorizer.transform(clean_test_data)
    np.asarray(test_data_features)

    print("Training the random forest")
    forest = RandomForestClassifier(n_estimators=30, n_jobs=1, verbose=3)
    forest = forest.fit(train_data_features, train_answers)

    print("Applying model to the test data")
    test_results = forest.predict(test_data_features)

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

main()
