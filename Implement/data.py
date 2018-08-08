import numpy as np
import pandas as pd
from sklearn.utils import shuffle


FNAME_pbeardat = "../Data/Polar_Bear_Adult_4_Geoff.csv"
FEATURES = ["d15N","d13C","d2H","d18O"]


def _int_labels(y):
    dic = {}
    cur = 0
    r = []
    for elem in y:
        if elem[0] not in dic:
            dic[elem[0]] = cur
            cur+=1
        r.append(dic[elem[0]])
    return r, dic


def _prep_bayesian_classifier(remove_outliers=True):
    OUTCOMES = ["ZONE"]
    df = pd.read_csv(FNAME_pbeardat)
    if remove_outliers:
        df = df.loc[df["Delete2"] == 0]
    df = df.dropna()
    df = shuffle(df)
    x, y = df[FEATURES].values, df[OUTCOMES].values
    y, zone_dic = _int_labels(y)
    return x, y, zone_dic


def prep(model="Bayesian Classifier"):
    try:
        return {
            "Bayesian Classifier":_prep_bayesian_classifier
        }[model]()
    except KeyError:
        raise ValueError("Invalid model {model}".format(model=model))