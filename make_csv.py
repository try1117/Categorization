import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
import math
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import csv

def create_categories_table():
    print("Reading file")
    noms = pd.read_csv("data/utf-8/nomenclature_utf8.csv")
    category_id = []
    category_name = []
    category_id_to_idx = {}

    print("Iterating")

    for index, row in noms.iterrows():
        if row["Vid"] == "Категория":
            category_id_to_idx[row["_IDRRef"]] = len(category_id)
            category_id += [row["_IDRRef"]]
            category_name += [row["_Description"]]
        if index % 1000 == 0:
            print(index / noms.shape[0] * 100)

    category_size = np.zeros((len(category_id),), dtype=int)
    print(len(category_size))

    valid_product_id = set()
    products = pd.read_csv("data/utf-8/products_dns_short_norp_utf8.csv")
    for index, row in products.iterrows():
        valid_product_id.add(row["product_id"])

    print("Id len = {}".format(len(valid_product_id)))

    prod_to_noms = pd.read_csv('data/utf-8/product_to_nomenclature_utf8.csv')
    for index, row in prod_to_noms.iterrows():
        if (index % 100000 == 0):
            print("Done {:.2f} %".format(index / prod_to_noms.shape[0] * 100))
        if (row["_Fld478RRef"] != "B953C2DB672F7B3148DEB2B798E86F9C"):
            continue
        cat = row["_Fld480_RRRef"]
        if (cat in category_id_to_idx) and (row["_Fld477RRef"] in valid_product_id):
            category_size[category_id_to_idx[cat]] += 1
            valid_product_id.remove(row["_Fld477RRef"])

    categories_data = {"category_id": category_id, "category_size": category_size, "category_name": category_name}
    categories = pd.DataFrame(categories_data, columns=["category_id", "category_size", "category_name"])
    categories = categories.sort_values(by=["category_size"], ascending=False)
    categories.to_csv("data/utf-8/categories_utf8.csv", sep=",", quoting=csv.QUOTE_ALL, index=False, encoding='UTF-8')

# create_categories_table()

def create_products_table():
    print("Reading file")
    prod_to_noms = pd.read_csv('data/utf-8/product_to_nomenclature_utf8.csv')
    prod_id_to_category_id = {}

    for index, row in prod_to_noms.iterrows():
        if (index % 100000 == 0):
            print("Done {:.2f} %".format(index / prod_to_noms.shape[0] * 100))
        if (row["_Fld478RRef"] == "B953C2DB672F7B3148DEB2B798E86F9C"):
            prod_id_to_category_id[row["_Fld477RRef"]] = row["_Fld480_RRRef"]

    print("Reading file")
    products = pd.read_csv("data/utf-8/products_dns_clear_utf8.csv")
    products.drop(["_ParentIDRRef", "_Folder", "_Code", "_Fld214", "_Description"], axis="columns", inplace=True)
    products.rename(columns={"_IDRRef": "product_id", "_Fld217": "description"}, inplace=True)
    products.insert(1, "category_id", np.empty(products.shape[0], dtype=str))

    unique_ids = set()
    for index, row in products.iterrows():
        if (row["product_id"] in prod_id_to_category_id):
            unique_ids.add(row["product_id"])

    print("Unique id = {} from {} = {:.3f} %".format(len(unique_ids), products.shape[0], len(unique_ids) / products.shape[0] * 100))

    data = {
        "product_id": np.empty(len(unique_ids), dtype=str),
        "category_id": np.empty(len(unique_ids), dtype=str),
        "description": np.empty(len(unique_ids), dtype=str),
    }
    products_no_rp = pd.DataFrame(data, columns=["product_id", "category_id", "description"])
    unique_ids = set()

    for index, row in products.iterrows():
        if (index % 100000 == 0):
            print("Done {:.2f} %".format(index / products.shape[0] * 100))
        if (row["product_id"] in prod_id_to_category_id) and (not row["product_id"] in unique_ids):
            products_no_rp.set_value(len(unique_ids), "product_id", row["product_id"])
            products_no_rp.set_value(len(unique_ids), "category_id", prod_id_to_category_id[row["product_id"]])
            products_no_rp.set_value(len(unique_ids), "description", row["description"])
            unique_ids.add(row["product_id"])
            # products.set_value(index, "category_id", prod_id_to_category_id[row["product_id"]])

    print(products_no_rp)
    products_no_rp.to_csv("data/utf-8/products_dns_short_norp_utf8.csv", sep=",", quoting=csv.QUOTE_ALL, index=False, encoding='UTF-8')

# create_products_table()

def create_sorted_products_table():
    print("Reading file")
    products = pd.read_csv("data/utf-8/products_dns_short_norp_utf8.csv")
    categories = pd.read_csv("data/utf-8/categories_utf8.csv")
    # print(categories)
    # quit()

    data = {
        "product_id": np.empty(products.shape[0], dtype=str),
        "category_id": np.empty(products.shape[0], dtype=str),
        "description": np.empty(products.shape[0], dtype=str),
    }
    sorted_products = pd.DataFrame(data, columns=["product_id", "category_id", "description"])

    sizes = {}
    for index, row in products.iterrows():
        if not row["category_id"] in sizes:
            sizes[row["category_id"]] = 1
        else:
            sizes[row["category_id"]] += 1

    sorted_by_value = sorted(sizes.items(), key=lambda kv: kv[1], reverse=True)
    # print(sorted_by_value)

    for index, row in categories.iterrows():
        if row["category_size"] != 0 and row["category_size"] != sorted_by_value[index][1]:
            print("{} instead of {}".format(row, sorted_by_value[index]))

    # print(lensum)
    # print(products.shape[0])
    quit()

    indices = {}
    lensum = 0
    for key, value in sorted_by_value:
        indices[key] = lensum
        lensum += value
        # print(row["category_size"])


    print("Iterating over products")
    for index, row in products.iterrows():
        if (row["category_id"] in indices):
            # print(indices[row["category_id"]])
            sorted_products.iloc[indices[row["category_id"]]] = row
            indices[row["category_id"]] += 1
        else:
            print(row["category_id"])

    print(sorted_products)
    sorted_products.to_csv("data/utf-8/products_dns_short_sorted_utf8.csv", sep=",", quoting=csv.QUOTE_ALL, index=False, encoding='UTF-8')

create_sorted_products_table()

# _Fld3041RRef,_Fld3042RRef,_Fld3043,_Fld3044RRef

# !!!!!!!!!!!
# DANGER ZONE
# working with 10 Gb table
# !!!!!!!!!!!

def create_competitors_table():
    print("Reading file")

    chunksize = 10 ** 6
    cnt = 0
    with open("data/competitors_utf8.csv", "w", encoding="UTF-8") as res_file:
        res_file.write("\"product_id\",\"description\"\n")
        for products_data in pd.read_csv("data/кон.csv", encoding="UTF-16", chunksize=chunksize):
            products_data.drop(["_Marked", "_OwnerIDRRef", "_Code", "_Description", "_Fld182"], axis="columns", inplace=True)
            products_data.to_csv(res_file, header=False, index=False, sep=",", quoting=csv.QUOTE_ALL, encoding="UTF-8")
            cnt += 1
            print("Chunk {:02d} processed".format(cnt))

def append_header():
    chunksize = 10 ** 6
    cnt = 0
    with open("data/competitors_utf8.csv", "w", encoding="UTF-8") as res_file:
        res_file.write("\"product_id\",\"description\"\n")
        with open("data/competitors_without_header_utf8.csv", "r", encoding="UTF-8") as data_file:
            while True:
                data = data_file.read(65536)
                if data:
                    res_file.write(data)
                else:
                    break

# append_header()

def view_competitors_file():
    with open("data/competitors_utf8.csv", "r", encoding="UTF-8") as file:
        for i in range(100):
            print(file.readline())

        chunksize = 100
        for products_data in pd.read_csv(data_file, encoding=BASIC_ENCODING, chunksize=chunksize):
            continue

        quit()

# view_competitors_file()
