from sklearn import datasets, svm, metrics
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit
import cPickle
import numpy as np 

fn = '/Users/graemecox/Documents/Capstone/Code/eegSvm/labels.txt'
UPDATE_FEAT = 0
CREATE_CLASSIFIER = 1
CLASSIFIER_NAME = 'savedClassifier.pkl'

def readLabels(root):
	labels = f.read()
	# For Labels
	labels = []
	with open(fn) as f:
		for line in f:
			#Split each line by comma, then save as a list of floats
			currentline = [float(x) for x in line.split(",")]
			labels.append(currentline)
	f.close()
	np.save('labels_py',np.array(labels))
	return labels
	

def readFeatures(root):
	# For features
	fn = '/Users/graemecox/Documents/Capstone/Code/eegSvm/features.txt'

	feat = []
	with open(fn) as f:
		for line in f:
			currentline = [float(x) for x in line.split(",")]
			feat.append(currentline)
	f.close()
	np.save('feat_py',np.array(feat))
	return feat


def createLinearClassifier(X_train, y_train):

	clf = svm.SVC()

	# Data stuff
	y_train.ravel()

	clf.fit(X_train,y_train)
	print("Classifier has been created")
	#Save classifier
	with open(CLASSIFIER_NAME,'wb') as fid:
		cPickle.dump(clf, fid)
	print("Classifier has been saved")
	return clf

def crossValidationStuff(model, X_test, y_test, cv_num):
	scores = cross_val_score(model, X_test, y_test, cv_num)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() *2))




# If we want to load from the text file, or load from numpy
if (UPDATE_FEAT):
	labels = readLabels(root)
	feat = readFeatures(root)
	print("Read in labels and features")
else:
	labels = np.load('labels_py.npy')
	feat = np.load('feat_py.npy')
	print("Loaded labels and features")


train_X, test_X, train_y, test_y = train_test_split(
	feat, labels, test_size=0.2, random_state=0)
c,r = train_y.shape
train_y = train_y.reshape(c,)
c,r = test_y.shape
test_y = test_y.reshape(c,)

# Create or load classifier

if (CREATE_CLASSIFIER):
	clf = createLinearClassifier(train_X, train_y)
	print('Created classifier')
else:
	with open(CLASSIFIER_NAME,'rb') as fid:
		clf = cPickle.load(fid)
	print('Loaded classifier')


## Look at scores





############################## OLD STUFF ###############################

# print(train_X)
# print(train_y.shape)
# print(test_X.shape)
# print(test_y.shape)


# # We'll use 500 of the samples just for testing
# train_X = feat[:-500]
# train_X = np.array(train_X)
# # print(train_X.shape)
# test_X = feat[:500]
# test_X = np.array(test_X)
# # print(test_X.shape)

# #Prep labels
# train_y = labels[:-500]
# train_y = np.array(train_y)
# train_y.ravel()
# # print(train_y)
# test_y = labels[:500]
# test_y = np.array(test_y)
# test_y.ravel()
# print(test_y.shape)
# testClf = svm.SVC(kernel='linear',C=1).fit(train_X, train_y)
# print(testClf.score(test_X, test_y))
# testClf = svm.SVC(kernel='linear',C=5).fit(train_X, train_y)
# print(testClf.score(test_X, test_y))
# testClf = svm.SVC(kernel='linear',C=10).fit(train_X, train_y)
# print(testClf.score(test_X, test_y))

## Train SVM




# lin_classifier = svm.LinearSVC()
# lin_classifier.fit(train_X,train_y)
# lin_score = lin_classifier.score(test_X,test_y)

# print("SVC Accuracy is %f\nLinear Classifier Accuracy is %f\n" %(clf_score,lin_score))




# #Prediction
# print(clf.predict(feat_pred))
# if clf.predict(feat_pred) == label_pred:
# 	print('WE PREDICTED CORRECTLY')
# else:
# 	print('STILL GOT SOME WORK TO DO')

# # iris = datasets.load_iris()
# # print(type(iris.data))
# # print(iris.data.size)


# ## Linear SVM
# lin_classifier = svm.LinearSVC()
# lin_classifier.fit(X,y)
# if lin_classifier.predict(feat_pred) == label_pred:
# 	print('WE PREDICTED CORRECTLY')
# else:
# 	print('STILL GOT SOME WORK TO DO')
