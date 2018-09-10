from abc import ABC, abstractmethod

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA

class Model(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def __init__(self, X, y, pca=False, n_components=3, k=5, print_cv=False):
        self.model = self.get_model()
        self.X, self.y, self.k = X, y, k
        if pca:
            self.X = PCA(n_components=n_components).fit_transform(self.X)
        self.cv_score = cross_val_score(self.model, self.X, self.y, cv=k)
        self.model.fit(self.X, self.y)
        if print_cv:
            print(self.model.best_params_)

    def cross_val_score(self):
        return cross_val_score(self.model, self.X, self.y, cv=self.k)
    
    def unstratified_score(self):
        return self.model.score(self.X,self.y)

class BayesClassifier(Model):

    def get_model(self):
        return GaussianNB()

    def predict(self, single):
        return self.model.predict([single]), self.model.predict_proba([single])

class SVM(Model):

    def get_model(self):
        return SVC(kernel='rbf', C=1)

class RForest(Model):

    def get_model(self):
        return RandomForestClassifier(
            bootstrap=True,
            min_samples_leaf=3,
            min_samples_split=2,
            n_estimators=40
        )

class KNeigh(Model):

    def get_model(self):
        return KNeighborsClassifier(
            p=2,
            algorithm='ball_tree',
            n_neighbors=20
        )

class FFNN(Model):

    def get_model(self):
        return MLPClassifier(
            activation='logistic',
            max_iter=5000,
            hidden_layer_sizes=(10,10))

class RidgeCV(Model):

    def get_model(self):
        return RidgeClassifierCV()