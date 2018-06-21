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

    prod_to_noms = pd.read_csv('data/utf-8/product_to_nomenclature_utf8.csv')
    for index, row in prod_to_noms.iterrows():
        if (index % 100000 == 0):
            print("Done {:.2f} %".format(index / prod_to_noms.shape[0] * 100))
        if (row["_Fld478RRef"] != "B953C2DB672F7B3148DEB2B798E86F9C"):
            continue
        cat = row["_Fld480_RRRef"]
        if cat in category_id_to_idx:
            category_size[category_id_to_idx[cat]] += 1

    categories_data = {"category_id": category_id, "category_size": category_size, "category_name": category_name}
    categories = pd.DataFrame(categories_data, columns=["category_id", "category_size", "category_name"])
    categories = categories.sort_values(by=["category_size"], ascending=False)
    categories.to_csv("data/utf-8/categories_utf8.csv", sep=",", quoting=csv.QUOTE_ALL, index=False, encoding='UTF-8')

def create_products_table():
    print("Reading file")
    prod_to_noms = pd.read_csv('data/utf-8/product_to_nomenclature_utf8.csv')
    prod_id_to_category_id = {}

    for index, row in prod_to_noms.iterrows():
        if (index % 100000 == 0):
            print("Done {:.2f} %".format(index / prod_to_noms.shape[0] * 100))
        #     break

        if (row["_Fld478RRef"] == "B953C2DB672F7B3148DEB2B798E86F9C"):
            prod_id_to_category_id[row["_Fld477RRef"]] = row["_Fld480_RRRef"]

        # if (row["_Fld477RRef"] == "877900155D03330D11E75D3C1633F2F7"):
        #     print("gocha")
        #     if (row["_Fld478RRef"] == "B953C2DB672F7B3148DEB2B798E86F9C"):
        #         print(row["_Fld480_RRRef"])
        #     else:
        #         print("error")
        #     # quit()

    print("Reading file")
    products = pd.read_csv("data/utf-8/products_dns_clear_utf8.csv")
    products = products.drop(["_ParentIDRRef", "_Folder", "_Code", "_Fld214", "_Fld217"], axis="columns")
    products.rename(columns={"_IDRRef": "product_id", "_Description": "description"}, inplace=True)
    products.insert(1, "category_id", np.empty(products.shape[0], dtype=str))

    # products.set_value(0, "category_id", "asdasdasd")
    # print(products.loc[0,])
    # print(products)
    # quit()

    # print(products.iloc[[82567]])
    # if ("877900155D03330D11E75D3C1633F2F7" in prod_id_to_category_id)
    # print(prod_id_to_category_id["877900155D03330D11E75D3C1633F2F7"])
    # # print(prod_id_to_category_id[products.iloc[[82567]]["category_id"]])
    # quit()


    for index, row in products.iterrows():
        # if (row["product_id"] == "877900155D03330D11E75D3C1633F2F7"):
        #     print("hello")
        #     if not(row["product_id"] in prod_to_noms):
        #         print("Some error")
        #     print(prod_to_noms[row["product_id"]])
        #     quit()

        if (index % 100000 == 0):
            print("Done {:.2f} %".format(index / products.shape[0] * 100))
        if (row["product_id"] in prod_id_to_category_id):
            print("success")
            products.set_value(index, "category_id", prod_id_to_category_id[row["product_id"]])

    print(products)
    products.to_csv("data/utf-8/products_dns_correct_utf8.csv", sep=",", quoting=csv.QUOTE_ALL, index=False, encoding='UTF-8')

create_products_table()

# _Fld3041RRef,_Fld3042RRef,_Fld3043,_Fld3044RRef


