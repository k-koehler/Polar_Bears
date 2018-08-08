import readline

from bayes import BayesClassifier

def _rev_lookup(val, mydict):
    return (list(mydict.keys())[list(mydict.values()).index(val)])

while True:
    FEATURES = ["d15N","d13C","d2H","d18O"]
    model = BayesClassifier()
    model_input = []
    for f in FEATURES:
        model_input.append(float(input("{f}: ".format(f=f))))
    y_pred, y_pred_prob = model.predict(model_input)
    print(""
    "Model predicts {y_pred} with {conf}% confidence.".format(
        y_pred=_rev_lookup(y_pred, model.zone_dic),
        conf=round(y_pred_prob[0][y_pred][0]*100, 4)))