# Kevin Koehler
# 7 April 2018

# Preprocess the data
# Make it suitable for keras/tf

import pandas as pd
import numpy as np

# constants here
FNAME_pbeardat = "../../Data/Polar_Bear_Adult_4_Geoff.csv"
FEATURES = ["d15N", "d13C", "d2H"]
OUTCOMES1 = ["Latitude", "Longitude"]
OUTCOMES2 = ["ZONE"]
OUTLIER_FLAGS = ['Delete2']

# Load csv file into memory


def _load_csv(fname, remove_outliers):
    if not remove_outliers:
        return pd.read_csv(fname)
    else:
        df = pd.read_csv(fname)
        df.filter(items=OUTLIER_FLAGS)
        return df


def convert(y_zone):
    zone_dic = {'LS': 0, 'BB': 1, 'WH': 2, 'DS': 3, 'FB': 4, 'VM': 5, 'KB': 6}
    return [zone_dic[x] for x in y_zone['ZONE']]

# reshapes the dataframe to an np.array suitable for tf


def _reshape(features, outcomes, df, zone=False):
    for feature in features:
        df = df[np.isfinite(df[feature])]
    if not zone:
        for outcome in outcomes:
            df = df[np.isfinite(df[outcome])]
    x = df.filter(items=features)
    y = df.filter(items=outcomes)
    if zone:
        y = convert(y)

    return np.array(x), np.array(y)

# returns an numpy array in suitable shape for tensorflow


def prepare(arg=None, remove_outliers=False):
    df = _load_csv(FNAME_pbeardat, remove_outliers)
    if(arg == 'zone'):
        return _reshape(FEATURES, OUTCOMES2,  df, zone=True)
    else:
        return _reshape(FEATURES, OUTCOMES1,  df)
