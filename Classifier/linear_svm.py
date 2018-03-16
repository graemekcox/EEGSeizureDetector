import sys

sys.path.insert(0, '/Users/graemecox/Documents/Capstone/Code/eegSvm/PreprocessingFunctions')
sys.path.insert(0, '/Users/graemecox/Documents/Capstone/Code/eegSvm/Features')

from fileReader import *

from sklearn import datasets, svm, metrics, tree, neighbors, neural_network
from sklearn import naive_bayes, linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process.kernels import RBF

import cPickle
import numpy as np 

# fn = '/Users/graemecox/Documents/Capstone/Code/eegSvm/Data/labels.txt'
# UPDATE_FEAT = 1
# CREATE_CLASSIFIER = 1
CLASSIFIER_NAME = 'savedClassifier.pkl'

CLASSIFIERS = [
	svm.SVC(kernel='linear', C=0.035),
	tree.DecisionTreeClassifier(max_depth=5),
	neighbors.KNeighborsClassifier(3),
	neural_network.MLPClassifier(alpha=1),
	naive_bayes.GaussianNB(),
	svm.LinearSVC(),
	LinearDiscriminantAnalysis(solver='svd', store_covariance=True),
	QuadraticDiscriminantAnalysis(store_covariances=True),
	# GaussianProcessClassifier(1.0*RBF(1.0)),
	RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
	AdaBoostClassifier()
	# tree.DecisionTreeClassifier(max_depth=5, n_estimators=10, max_features=1)
	]
# root = '/Users/graemecox/Documents/Capstone/Code/eegSvm/'




def createLinearClassifier(X_train, y_train,save=0):

	model = svm.SVC(C=25, kernel='rbf',degree=1)

	# Data stuff
	y_train.ravel()

	model.fit(X_train,y_train)
	print("Classifier has been created")

	if save:
	#Save classifier
		with open(CLASSIFIER_NAME,'wb') as fid:
			cPickle.dump(model, fid)
		print("Classifier has been saved")

	return model

def crossValidationStuff(model, X_test, y_test, cv_num):
	scores = cross_val_score(model, X_test, y_test, cv=cv_num)
	print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() *2))



def train_multiple_classifiers(X_train, y_train, X_test, y_test, save=0):
	model  = []
	scores = []
	for i in range(len(CLASSIFIERS)):
		
		temp_model = CLASSIFIERS[i]
		print('%%%%%%%%%%%%  Classifier %d: %s' % ((1+i),CLASSIFIERS[i]))

		temp_model.fit(X_train, y_train)
		scores.append(temp_model.score(X_test,y_test))

		print('Score: %f' % scores[i])

	print('------------ FINISHED -----------')
	return scores

# labels = np.load('../Data/labels.npy')
# feat = np.load('../Data/features.npy')
# X_train, X_test, y_train, y_test = train_test_split(
# feat, labels, test_size=0.2, random_state=0)

# scores = train_multiple_classifiers(X_train, y_train, X_test, y_test,0)
# scores = np.array(scores)
# print('Classifier With the Highest Score: %s \nScore was: %s' % 
# 	(CLASSIFIERS[scores.argmax()], scores[scores.argmax()]))
