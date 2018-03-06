import sys

sys.path.insert(0, '/Users/graemecox/Documents/Capstone/Code/eegSvm/PreprocessingFunctions')
sys.path.insert(0, '/Users/graemecox/Documents/Capstone/Code/eegSvm/Features')

from fileReader import *

from sklearn import datasets, svm, metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit
import cPickle
import numpy as np 

fn = '/Users/graemecox/Documents/Capstone/Code/eegSvm/Data/labels.txt'
UPDATE_FEAT = 1
CREATE_CLASSIFIER = 1
CLASSIFIER_NAME = 'savedClassifier.pkl'
root = '/Users/graemecox/Documents/Capstone/Code/eegSvm/'


def createLinearClassifier(X_train, y_train):

	model = svm.SVC(C=25, kernel='rbf',degree=1)

	# Data stuff
	y_train.ravel()

	model.fit(X_train,y_train)
	print("Classifier has been created")
	#Save classifier
	with open(CLASSIFIER_NAME,'wb') as fid:
		cPickle.dump(model, fid)
	print("Classifier has been saved")
	return model

def crossValidationStuff(model, X_test, y_test, cv_num):
	scores = cross_val_score(model, X_test, y_test, cv=cv_num)
	print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() *2))





# If we want to load from the text file, or load from numpy
if (UPDATE_FEAT):
	# labels = readLabels(root)
	# feat = readFeatures(root)
	labels,feat = readCsv('../Data/features.csv',0)

	# labels = readTextFile('Data/labels.txt',1, 'labels_npy')
	# labels = readTextFile('Data/labels.txt',1, 'features_npy')
	print("Read in labels and features")
else:
	labels = np.load('Data/labels_py.npy')
	feat = np.load('Data/feat_py.npy')
	print("Loaded labels and features")


train_X, test_X, train_y, test_y = train_test_split(
	feat, labels, test_size=0.2, random_state=0)

## If data is not 1D array
# c,r = train_y.shape
# train_y = train_y.reshape(c,)
# c,r = test_y.shape
# test_y = test_y.reshape(c,)

# Create or load classifier

if (CREATE_CLASSIFIER):
	print('Creating classifier now. Sit tight!')
	clf = createLinearClassifier(train_X, train_y)
	print('Created classifier')
else:
	with open(CLASSIFIER_NAME,'rb') as fid:
		clf = cPickle.load(fid)
	print('Loaded classifier')


print(test_X.shape)
print(test_y.shape)
## Look at scores
# scores = cross_vahttps://gfycat.com/BlandBarrenCoatil_score(clf, test_X, test_y, cv=25)
# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() *2))

cv_num = ShuffleSplit(n_splits = 3, test_size=0.3, random_state=0)
crossValidationStuff(clf, test_X, test_y,cv_num)

