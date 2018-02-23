from sklearn import datasets, svm
import numpy as np 

fn = '/Users/graemecox/Documents/Capstone/Code/eegSvm/labels.txt'

# labels = f.read()
## For Labels
labels = []
with open(fn) as f:
	for line in f:
		#Split each line by comma, then save as a list of floats
		currentline = [float(x) for x in line.split(",")]
		labels.append(currentline)
label_pred = labels[0]

del labels[0]
# For features
fn = '/Users/graemecox/Documents/Capstone/Code/eegSvm/features.txt'

feat = []
with open(fn) as f:
	for line in f:
		currentline = [float(x) for x in line.split(",")]
		feat.append(currentline)
f.close()
feat_pred = feat[0]
del feat[0]
# ## First line is a header line
# del feat[0]


#Prep data
# We'll use 500 of the samples just for testing
train_X = feat[:-500]
train_X = np.array(train_X)
print(train_X.shape)
test_X = feat[:500]
test_X = np.array(test_X)
print(test_X.shape)

#Prep labels
train_y = labels[:-500]
train_y = np.array(train_y)
train_y.ravel()
print(train_y)
test_y = labels[:500]
test_y = np.array(test_y)
print(test_y.shape)



## Train SVM


clf = svm.SVC()

clf.fit(train_X,train_y)
clf_score = clf.score(test_X, test_y)

lin_classifier = svm.LinearSVC()
lin_classifier.fit(train_X,train_y)
lin_score = lin_classifier.score(test_X,test_y)

print("SVC Accuracy is %f\nLinear Classifier Accuracy is %f\n" %(clf_score,lin_score))
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
