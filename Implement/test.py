from data import prep
from models import BayesClassifier, SVM, RForest, KNeigh, FFNN, RidgeCV


def test_model(Classifier, prep_type, **clfargs):
    X, y, *_ = prep(prep_type)
    clf = Classifier(X=X, y=y, **clfargs)
    print("No PCA, 5-fold cross-val score...")
    print("Score = {score}% ({scores})".format(score=round(clf.cross_val_score().mean(), 4)*100, scores=clf.cv_score))
    print("No PCA, unstratified score...")
    print("Score = {score}%".format(score=round(clf.unstratified_score(),4)*100))
    clf = Classifier(X=X, y=y, pca=True, n_components=4, **clfargs)
    print("PCA, 5-fold-cross-val score")
    print("Score = {score}% ({scores})".format(score=round(clf.cv_score.mean(), 4)*100, scores=clf.cv_score))
    
print("BAYES")
test_model(Classifier=BayesClassifier, prep_type="basic")
print("SVM")
test_model(Classifier=SVM, prep_type="scale")
print("RANDOM FOREST")
test_model(Classifier=RForest, prep_type="basic")
print("KNEIGHBOURS")
test_model(Classifier=KNeigh, prep_type="scale")
print("MLP")
X, y, *_ = prep("basic")
clf = FFNN(X=X, y=y, pca=True, n_components=4)
print("PCA, 5-fold-cross-val score")
print("Score = {score}% ({scores})".format(score=round(clf.cv_score.mean(), 4)*100, scores=clf.cv_score))
print("RIDGE")
test_model(Classifier=RidgeCV, prep_type="scale")