import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

FNAME_pbeardat = "../Data/Polar_Bear_Adult_4_Geoff.csv"
FEATURES = ["d15N","d13C","d2H","d18O"]
OUTCOME = "ZONE"

def _basic_prep(remove_outliers=True):
    df = pd.read_csv(FNAME_pbeardat)
    if remove_outliers:
        df = df.loc[df["Delete2"] == 0]
    df = df.dropna()
    df = shuffle(df)
    df[OUTCOME], zone_dic = pd.factorize(df[OUTCOME])
    X, y = df[FEATURES].values, df[OUTCOME].values
    return X, y, zone_dic

def _prep_bayesian_classifier(remove_outliers=True):
    return _basic_prep(remove_outliers=remove_outliers)

def _prep_svm():
    X,y,labels = _basic_prep(remove_outliers=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X=X)
    return X,y,labels

def prep(model="Bayesian Classifier"):
    try:
        return {
            "basic":_prep_bayesian_classifier,
            "scale":_prep_svm
        }[model]()
    except KeyError:
        raise ValueError("Invalid model {model}".format(model=model))