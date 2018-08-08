from bayes import BayesClassifier

clf = BayesClassifier()
print("No PCA, 5-fold cross-val score...")
print("Score = {score}% ({scores})".format(score=round(clf.cv_score.mean(), 4)*100, scores=clf.cv_score))
clf = BayesClassifier(pca=True, n_components=4)
print("PCA, 5-fold-cross-val score")
print("Score = {score}% ({scores})".format(score=round(clf.cv_score.mean(), 4)*100, scores=clf.cv_score))
print("No PCA, unstratified score...")
clf = BayesClassifier()
print("Score = {score}%".format(score=round(clf.unstratified_score(),4)*100))