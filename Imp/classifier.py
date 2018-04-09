import data as d

import tensorflow as tf
import numpy
import pandas

from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def add_dense_layers(model, n, width):
  for _ in range(0, n):
    model.add(Dense(width, activation = 'relu'))

def create_model(x, y):
  model = Sequential()
  model.add(Dense(3, input_shape=x.shape[1:], activation='relu'))
  add_dense_layers(model, 2, 10000)
  model.add(Dense(7, activation='softmax'))
  model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

zone_dic = {'LS':0, 'BB':1, 'WH':2, 'DS':3, 'FB':4, 'VM':5, 'KB':6 }
x, y = d.prepare('zone')
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1, random_state=42)
model = create_model(x_train,y_train)
print(model.summary())
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=32)
scores = model.evaluate(x_test, y_test, verbose=0)
print("Scores: ", scores)
print("Accuracy: %.2f%%" % (scores[1]*100))
