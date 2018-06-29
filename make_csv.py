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
import gc

def fill_sub_mega_table(file_opp_products, file_opp_to_dns):
    opp_to_dns_data = pd.read_csv(file_opp_to_dns, encoding="UTF-8")
    chunksize = 3 * (10 ** 5)
    chunk_idx = -1
    descriptions_done = 0

    opp_to_dns_data["opp_product_description"] = np.empty(opp_to_dns_data.shape[0], dtype=str)
    # print(opp_to_dns_data["opp_product_description"])
    # quit()

    for products_data in pd.read_csv(file_opp_products, encoding="UTF-8", chunksize=chunksize):
        chunk_idx += 1
        print("Block {:02d}".format(chunk_idx))
        id_to_desc = {}
        for index, row in products_data.iterrows():
            id_to_desc[row["product_id"]] = row["description"]
        print("Descriptions are in dictionary")

        for index, row in opp_to_dns_data.iterrows():
            if (row["opp_product_id"] in id_to_desc):
                opp_to_dns_data.set_value(index, "opp_product_description", id_to_desc[row["opp_product_id"]])
                descriptions_done += 1

        print("Total descriptions found {}".format(descriptions_done))
        opp_to_dns_data.to_csv("data/mega_sub_1e6_filled_{:02d}.csv".format(chunk_idx), index=False, sep=",", quoting=csv.QUOTE_ALL, encoding="UTF-8")
        if (descriptions_done == opp_to_dns_data.shape[0]):
            print("Found all descriptions")
            quit()

fill_sub_mega_table("data/competitors_utf8.csv", "data/mega_subtable_1000000.csv")
quit()

def create_sub_mega_table(file_opp_to_dns, records):
    for data in pd.read_csv(file_opp_to_dns, encoding="UTF-8", chunksize=records):
        # cat = set()
        # for index, row in data.iterrows():
        #     cat.add(row["category_id"])

        # print(len(cat))
        data.to_csv("data/mega_subtable_{}.csv".format(records), index=False, sep=",", quoting=csv.QUOTE_ALL, encoding="UTF-8")
        break

# create_sub_mega_table("data/opp_to_dns_cat_desc_part_utf8.csv", 1000000)
# quit()

def check_opp_desc(file_opp_to_dns):
    chunksize = 10 ** 6
    descriptions_cnt = 0
    idx = -1
    for products_data in pd.read_csv(file_opp_to_dns, encoding="UTF-8", chunksize=chunksize):
        idx += 1
        print("Block {}".format(idx))
        if (idx == 0):
            continue
        for index, row in products_data.iterrows():
            if (not pd.isnull(row["opp_product_description"])):
                descriptions_cnt += 1
        print(descriptions_cnt)

# check_opp_desc("data/opp_to_dns_cat_desc_part_utf8_00.csv")
# quit()

def mega_table_add_opp_desc(file_opp_products, L, R, file_opp_to_dns):
    chunksize = 10 ** 6
    opp_idx = -1
    prev_file_opp_to_dns = file_opp_to_dns
    new_descriptions_cnt = 0

    for products_data in pd.read_csv(file_opp_products, encoding="UTF-8", chunksize=chunksize):
        opp_idx += 1
        print("---OPPONENT BLOCK {:02d}---".format(opp_idx))
        if (opp_idx < L):
            print("Skip this block")
            continue
        if (opp_idx > R):
            print("Skip all remaining blocks")
            break

        id_to_desc = {}
        for index, row in products_data.iterrows():
            id_to_desc[row["product_id"]] = row["description"]
        print("Descriptions are in dictionary")

        next_file_opp_to_dns = file_opp_to_dns + "__" + "{:02d}".format(opp_idx)
        with open(next_file_opp_to_dns, "w", encoding="UTF-8") as res_file:
            res_file.write('"opp_product_id","opp_product_description","dns_product_id","dns_product_description","category_id","type"\n')

            inner_block_idx = -1
            for opp_to_dns_data in pd.read_csv(prev_file_opp_to_dns, encoding="UTF-8", chunksize=chunksize):
                for index, row in opp_to_dns_data.iterrows():
                    if row["opp_product_id"] in id_to_desc:
                        products_data.set_value(index, "opp_product_description", id_to_desc[row["opp_product_id"]])
                        new_descriptions_cnt += 1

                opp_to_dns_data.to_csv(res_file, header=False, index=False, sep=",", quoting=csv.QUOTE_ALL, encoding="UTF-8")
                inner_block_idx += 1
                print("Inner chunk {:02d} processed".format(inner_block_idx))

        print("Total new descriptions added {}".format(new_descriptions_cnt))
        prev_file_opp_to_dns = next_file_opp_to_dns
        print("---OPPONENT BLOCK {:02d} PROCESSED---".format(opp_idx))

# mega_table_add_opp_desc("data/competitors_utf8.csv", 0, 0, "data/opp_to_dns_cat_desc_part_utf8.csv")
# quit()

def opp_to_dns_add_cat_and_desc(file_dns_products, file_opp_to_dns):
    print("Reading {}".format(file_dns_products))
    dns_products = pd.read_csv(file_dns_products, encoding='UTF-8')
    print(dns_products.shape)
    print(dns_products.shape[0])

    print("Put pairs (product_id, category_id) to the map")
    dns_id_to_info = {}
    for index, row in dns_products.iterrows():
        dns_id_to_info[row["product_id"]] = [row["category_id"], row["description"]]

    del dns_products
    gc.collect()
    print("Dictionary length = {}".format(len(dns_id_to_info)))

    chunksize = 3 * (10 ** 5)
    chunk_idx = 0
    with open("data/opp_to_dns_cat_desc_part_utf8.csv", "w", encoding="UTF-8") as res_file:
        res_file.write('"opp_product_id","opp_product_description","dns_product_id","dns_product_description","category_id","type"\n')
        for products_data in pd.read_csv(file_opp_to_dns, encoding="UTF-8", chunksize=chunksize):
            cnt = products_data.shape[0]
            products_data.insert(1, "opp_product_description", np.empty(cnt, dtype=str))
            products_data.insert(3, "dns_product_description", np.empty(cnt, dtype=str))
            products_data.insert(4, "category_id", np.empty(cnt, dtype=str))

            for index, row in products_data.iterrows():
                id = row["dns_product_id"]
                products_data.set_value(index, "category_id", dns_id_to_info.get(id, ["", ""])[0])
                products_data.set_value(index, "dns_product_description", dns_id_to_info.get(id, ["", ""])[1])

            products_data.to_csv(res_file, header=False, index=False, sep=",", quoting=csv.QUOTE_ALL, encoding="UTF-8")
            chunk_idx += 1
            print("Chunk {:02d} processed".format(chunk_idx))

    # opp_to_dns = pd.read_csv(file_opp_to_dns, encoding='UTF-8')
    # opp_to_dns_categories = np.empty(opp_to_dns.shape[0], dtype=str)

    # for index, row in opp_to_dns.iterrows():
        # opp_to_dns_categories[index] = dns_prod_to_cat.get(row["dns_product_id"], "")

    # opp_to_dns.insert(2, "category_id", opp_to_dns_categories)
    # opp_to_dns.to_csv("data/opp_to_dns_categories_utf8.csv", sep=",", quoting=csv.QUOTE_ALL, index=False, encoding='UTF-8')

# opp_to_dns_add_cat_and_desc("data/utf-8/products_dns_unique_blank_cat_utf8.csv", "data/opp_to_dns_utf8.csv")
# quit()

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

def create_unique_blank_cat_table():
    print("Reading file")
    products = pd.read_csv('data/utf-8/products_dns_correct_utf8.csv')

    unique_ids = set()
    for index, row in products.iterrows():
        if (index % 100000 == 0):
            print("Done {:.2f} %".format(index / products.shape[0] * 100))

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

        if (not row["product_id"] in unique_ids):
            products_no_rp.iloc[len(unique_ids)] = row
            unique_ids.add(row["product_id"])

    products_no_rp.to_csv("data/utf-8/products_dns_unique_blank_cat_utf8.csv", sep=",", quoting=csv.QUOTE_ALL, index=False, encoding='UTF-8')

# create_unique_blank_cat_table()
# quit()

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
        # if (row["product_id"] in prod_id_to_category_id) and (not row["product_id"] in unique_ids):
        if (not row["product_id"] in unique_ids):
            products_no_rp.set_value(len(unique_ids), "product_id", row["product_id"])
            products_no_rp.set_value(len(unique_ids), "category_id", prod_id_to_category_id.get(row["product_id"], ""))
            products_no_rp.set_value(len(unique_ids), "description", row["description"])
            unique_ids.add(row["product_id"])
            # products.set_value(index, "category_id", prod_id_to_category_id[row["product_id"]])

    # print(products_no_rp)
    products_no_rp.to_csv("data/utf-8/products_dns_unique_blank_cat_utf8.csv", sep=",", quoting=csv.QUOTE_ALL, index=False, encoding='UTF-8')

# create_products_table()
# quit()

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
