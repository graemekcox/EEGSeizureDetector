import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt

def crossValidationStuff(model, X_test, y_test, cv_num):
	scores = cross_val_score(model, X_test, y_test, cv=cv_num)
	print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() *2))
	return scores


def computeStats(model, X_test, y_train, y_test,cv = 5):

	score = crossValidationStuff(model, X_test, y_test, cv=cv)
	# score = crossValidationStuff(model, X_test, y_test, cv=5)
	# probs = model.predict_proba(X_test)
	# # preds = probs[:,1]
	# print(probs.shape)


def classifierStats(clf, X_test, y_test):
	pred = clf.predict(X_test) # get list of predictions for comparison

	print('----- Printing Classification Report ----- \n%s'%metrics.classification_report(y_test, pred, target_names=['Ictal','Interictal']))
	

	print('----- Confusion Matrix ------\n%s'%metrics.confusion_matrix(y_test, pred))

	tn,fp,fn,tp = metrics.confusion_matrix(y_test,pred).ravel()
	print('True Postivies: %f\t False Positives: %f\n' % (tp,fp))
	print('True Negatives: %f\t False Negatives: %f\n' % (tn,fn))
	tp = float(tp)
	tn = float(tn)
	print(tp)
	print(tn)
	specificity = tn/(tn+fp)
	print (specificity)
	selectivity = tp/(tp+fn)
	print ('The specificity is %f and the selectivity is %f',specificity,selectivity)
	return specificity,selectivity

def scoreUsingKaggle(fn_list):

	"""
	Score test data to input to kaggle to get a proper score.

	"""

	test_size = len(fn_list)

def getConfusionMatrix(results, expected):
	tn,fp,fn,tp = metrics.confusion_matrix(expected,results).ravel()
	# print (tn,fp,fn,tp)
	print('True Postivies: %f\t False Positives: %f\n' % (tp,fp))
	print('True Negatives: %f\t False Negatives: %f\n' % (tn,fn))
	tp = float(tp)
	tn = float(tn)
	print(tp)
	print(tn)
	specificity = tn/(tn+fp)
	print (specificity)
	selectivity = tp/(tp+fn)
	print ('The specificity is %f and the selectivity is %f',specificity,selectivity)
	return specificity,selectivity
