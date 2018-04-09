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

def pretty_compare(y, y_pred):
  c = np.column_stack((y, y_pred))
  lats = []
  longs = []
  for row in c:
    print row
    print "dif lat ", abs(row[0] - row[2]), " dif long ", abs(row[1] - row[3])
    lats.append(abs(row[0] - row[2]))
    longs.append(abs(row[1] - row[3]))
  print "avg lat err on test ", sum(lats)/len(lats)
  print "avg long err on test ", sum(longs)/len(longs)

#constants
WIDTH = 14
DEPTH = 3
EPOCHS = 600

def create_network():
	model = Sequential()
	model.add(Dense(3, input_shape=(3,), kernel_initializer='normal', activation='relu'))
	for _ in range(0, DEPTH):
		model.add(Dense(WIDTH, activation='relu'))
	model.add(Dense(2, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

seed = 42

x, y = d.prepare()
#print x, y
estimator = KerasRegressor(build_fn=create_network, epochs=EPOCHS, batch_size=100, verbose=0)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, x, y, cv=kfold)
print results.std()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state=3345)
estimator.fit(x_train, y_train)
prediction = estimator.predict(x_test)
pretty_compare(y_test, prediction)

