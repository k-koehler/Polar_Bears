import data as d
import random as random

import numpy as np
np.set_printoptions(threshold=np.inf)

from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.layers import Dense
from keras.models import Sequential
import numpy
import pandas
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import geopy.distance

#constants
WIDTH = 100
DEPTH = 5
EPOCHS = 100
SEED1 = 42
SEED2 = 13
N_SPLITS = 10
TEST_THRESH = 0.1
BATCH_SIZE = 32
REMOVE_OUTLIERS = True
FLEN = len(d.FEATURES)
FSHAPE = (FLEN, )

def pretty_compare(y, y_pred):
  c = np.column_stack((y, y_pred))
  for dp in c:
    print(dp)
    print("dist in km ", geopy.distance.vincenty( (dp[0],dp[1]) , (dp[2], dp[3])).km)

def create_network():
	model = Sequential()
	model.add(Dense(FLEN, input_shape=FSHAPE, kernel_initializer='normal', activation='relu'))
	for _ in range(0, DEPTH):
		model.add(Dense(WIDTH, activation='relu'))
	model.add(Dense(2, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

x, y = d.prepare(remove_outliers=REMOVE_OUTLIERS)
estimator = KerasRegressor(build_fn=create_network, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
kfold = KFold(n_splits=N_SPLITS, random_state=SEED1)
results = cross_val_score(estimator, x, y, cv=kfold)
print("mean squared error: ", results.std())

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=TEST_THRESH, random_state=SEED2)
estimator.fit(x_train, y_train)
prediction = estimator.predict(x_test)
pretty_compare(y_test, prediction)

