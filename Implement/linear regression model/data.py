#Kevin Koehler
#7 April 2018

#Preprocess the data
#Make it suitable for keras/tf

import pandas as pd
import numpy as np

#constants here
FNAME_pbeardat = "../../Data/Modified.csv"
FEATURES = ["YEAR"]
OUTCOMES1 = ["Latitude","Longitude"]
OUTCOMES2 = ["ZONE"]
OUTCOMES3 = ["Longitude"]
TEST_PERCENT = 0.1

#Load csv file into memory
def _load_csv(fname):
	return pd.read_csv(fname)

def convert(y_zone):
  zone_dic = {'LS':0, 'BB':1, 'WH':2, 'DS':3, 'FB':4, 'VM':5, 'KB':6, 'SB':7}
  return [zone_dic[x] for x in y_zone['ZONE']]

#reshapes the dataframe to an np.array suitable for tf
def _reshape(features, outcomes, test_thresh, df, zone=False):
  df = df.loc[df['Delete2'] == 0]
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
	
#returns an numpy array in suitable shape for tensorflow
def prepare(arg=None):
	df = _load_csv(FNAME_pbeardat)
	if(arg == 'zone'):
		return _reshape(FEATURES, OUTCOMES2, TEST_PERCENT, df, zone=True)
	else:
		return _reshape(FEATURES, OUTCOMES3, TEST_PERCENT, df)
