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

def opponents_to_dns_to_normal_csv(file_name):
    data = pd.read_csv(file_name, encoding='UTF-16')
    data.drop(["_Fld3044RRef"], axis="columns", inplace=True)
    data.rename(columns={"_Fld3041RRef": "opp_product_id", "_Fld3042RRef": "dns_product_id", "_Fld3043": "type"}, inplace=True)
    data.to_csv(file_name[:-4] + "_utf8.csv", sep=",", quoting=csv.QUOTE_ALL, index=False, encoding='UTF-8')
    print(file_name + " done")

opponents_to_dns_to_normal_csv("data/opp_to_dns.csv")
quit()

def to_normal_csv(file_name):
    data = pd.read_csv(file_name, encoding='UTF-16')
    data.to_csv(file_name[:-4] + "_utf8.csv", sep=",", quoting=csv.QUOTE_ALL, index=False, encoding='UTF-8')
    print(file_name + " done")

def process_products_file(file_name):
    data = pd.read_csv(file_name, encoding='UTF-16')

    print("Size before: {}".format(data.shape[0]))
    data = data[data._Folder != 0]
    print("Size after: {}".format(data.shape[0]))

    data.to_csv(file_name[:-4] + "_clear_utf8.csv", sep=",", quoting=csv.QUOTE_ALL, index=False, encoding='UTF-8')
    print(file_name + " done")

# to_normal_csv("data/my/nomenclature.csv")
# to_normal_csv("data/my/product_nomenclature.csv")
process_products_file("data/my/products_dns.csv")
quit()

def good_line(line, comma_cnt):
    if (line[0] != "\"" or line[0] != "\"" or line.count(',') != comma_cnt):
        return False

    s = line[1:73]
    if (re.match('^([a-zA-Z0-9]{32}\,[0-9]{2}\,){2}$', s)):
        return True
    return False;

if (False):
    f = open("1.csv", "r", encoding="UTF-16")
    out = open("1_valid.csv", "w", encoding="UTF-16")

    fl = f.readline()
    comma_cnt = fl.count(',')
    out.write(fl[1:-2] + "\n")

    for line in f:
        if (line[0] == "\"" and line[0] == "\"" and line.count(',') == comma_cnt):
            if (good_line(line, comma_cnt)):
                out.write(line[1:-2] + "\n")
            # out.write(line[1:-2] + "\n")

    quit()

data = pd.read_csv("1_valid.csv", sep=',', encoding="UTF-16", header=0) #, delimiter=',', quoting=3)

if False:
    # print(len(set(data["_ParentIDRRef"][:])))
    cat_data = data.drop(["_Marked", "_Folder", "_Code", "_Fld214", "_Fld217", "_Fld222", "_Fld223", "_Fld225"], axis="columns")
    cat_data = cat_data.groupby("_ParentIDRRef").count()
    cat_data = cat_data.sort_values("_IDRRef", ascending=False)

    # print(cat_data.shape[0])
    indices = cat_data.index.tolist()
    cnt = 0

    # cat_data.loc[:, "_Description_2"] = pd.Series(0, index=cat_data.index)
    # cat_data.loc[:, "_Description_3"] = pd.Series(0, index=cat_data.index)

    for i in indices:
        tmp = data[data["_ParentIDRRef"] == i].iloc[:3]

        cat_data.loc[i, "_Description"] = tmp.iloc[0]["_Description"]
        # cat_data.loc[i, "_Description_2"] = tmp.iloc[1]["_Description"]
        # cat_data.loc[i, "_Description_3"] = tmp.iloc[2]["_Description"]

        cnt += 1
        if (cnt % 200 == 0):
            print(cnt / 1000)
        if (cnt == 1000):
            break

    print(cat_data)
    # quit()
    # print(cat_data)
    cat_data.to_csv("categories_1000.csv", quoting=1, encoding="UTF-16")
else:
    cur = data[data["_ParentIDRRef"] == "83ED005056A4699B11E4389160160FA3"]
    cur = cur.drop(["_Marked", "_Folder", "_Code", "_Fld214", "_Fld217", "_Fld222", "_Fld223", "_Fld225"], axis="columns")
    print(cur)
