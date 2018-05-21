import data as d
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler

def multiple_linear_regression_model(x,y):
  print("multiple linear regression");
  model = LinearRegression(fit_intercept = True, normalize = True)
  mse = cross_validation.cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=10)
  print("mse: " , mse.mean())
  r2 = cross_validation.cross_val_score(model, x, y, scoring='r2', cv=10)
  print("r2: " , r2.mean())
  model.fit(x,y)
  compare = [(int(round(x)),y) for (x,y) in zip(model.predict(x), y)]
  #for c in compare:
  #  print(c)
  print("accuracy: " , accuracy(compare))
  
def multivariate_linear_regression_model(x,y):
  print("multivariate linear regression");
  model = LinearRegression(fit_intercept = True, normalize = True)
  mse = cross_validation.cross_val_score(model, x, y, scoring='neg_mean_squared_error', cv=10)
  print("mse: " , mse.mean())
  r2 = cross_validation.cross_val_score(model, x, y, scoring='r2', cv=10)
  print("r2: " , r2.mean())
  #model.fit(x,y)
  #compare = [(x,y) for (x,y) in zip(model.predict(x), y)]
  #for c in compare:
  #  print(c)
  
def accuracy(compare):
  return len([(x,y) for (x,y) in compare if x==y])/len(compare)
  
  
x, y = d.prepare('zone')
x = StandardScaler().fit_transform(x)
multiple_linear_regression_model(x,y)

x, y = d.prepare()
x = StandardScaler().fit_transform(x)
multivariate_linear_regression_model(x,y)









