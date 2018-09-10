import readline, sys, pandas, numpy as np

from bayes import BayesClassifier
from data import FEATURES

if len(sys.argv) != 2:
    print("Invalid usage. Try ppd <filename>.")

#load the data
FNAME = str(sys.argv[1])
df = pandas.read_csv(FNAME)
df = df.dropna()

#load the classifier
model = BayesClassifier()

#classify the file
predictions = model.model.predict(df[FEATURES].values)
prediction_vals = model.model.predict_proba(df[FEATURES].values)
rev_zone_dic = {v:k for k,v in model.zone_dic.items()}
prediction_vals = np.array(list(map(lambda pv: list(map(lambda v: v*100, pv)), prediction_vals)))
output = np.column_stack((df['ID'].values, prediction_vals))
df = pandas.DataFrame(output)
df.columns = ["ID"] + [v for k,v in rev_zone_dic.items()]
df.to_csv("output.csv")
print("Results in ./output.csv")