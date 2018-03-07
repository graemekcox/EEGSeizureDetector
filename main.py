import sys

sys.path.insert(0, 'PreprocessingFunctions')


from kaggleReader import *
from sklearn import datasets, svm, metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit
import numpy as np

root = '/Users/graemecox/Documents/Capstone/Data/EEG_Data/'

feat,labels, test_clips = readKaggleDataset(root)

print(feat.shape)
print(labels.shape)
X_train, X_test, y_train, y_test = train_test_split(
	feat, labels, test_size=0.2, random_state=0)

model = svm.SVC(C=25, kernel='rbf',degree=1)

# Data stuff
y_train.ravel()

model.fit(X_train,y_train)

scores = cross_val_score(model, X_test, y_test, cv=5)
print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() *2))