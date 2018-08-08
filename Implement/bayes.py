from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import PCA

import data

class BayesClassifier:


    def __init__(self, pca=False, n_components=3, k=5):
        self.model = GaussianNB()
        self.x, self.y, self.zone_dic = data.prep()
        if pca:
            self.x = PCA(n_components=n_components).fit_transform(self.x)
        self.cv_score = cross_val_score(self.model, self.x, self.y, cv=k)
        self.model.fit(self.x, self.y)


    def unstratified_score(self):
        return self.model.score(self.x,self.y)
    

    def predict(self, single):
        return self.model.predict([single]), self.model.predict_proba([single])