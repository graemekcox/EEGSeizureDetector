## feature selection script
import numpy as np 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif,mutual_info_classif
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn import svm


def use_pca(features, numComp=3):
	pca = PCA(n_components=numComp)
	fit = pca.fit(features)

	print('PCA Explained Variance: %s' % fit.explained_variance_ratio_)
	plt.semilogy(fit.explained_variance_ratio_,'--o')
	plt.show()
	return fit.components_


def univariateSelection(features,labels):

	best_mean = 0
	best_test = ''
	best_numFeat = 0

	test_i = -1
	score = np.zeros((19,3), dtype='float')

	for test in [chi2,f_classif,mutual_info_classif]: #Iterate through various scoring tests

		test_i = test_i +1 
		print('----- Now running test %s'%test)
		for i in range(1,features.shape[1]):
			print('------ Feature Selection using k=%d-----'% i)
			model = svm.SVC(C=25, kernel='rbf',degree=1)

			k_feat = SelectKBest(score_func=test, k=i).fit_transform(features,labels)
			# k_fit =k_clf.fit(features,labels) 
			# print(k_fit.scores_) #Pick the ones with highest score
			# k_feat = k_fit.transform(features)
			print(k_feat.shape)

			X_train, X_test, y_train, y_test = train_test_split(
				k_feat, labels, test_size=0.2, random_state=0)

			model.fit(X_train, y_train)
	 

			scores = cross_val_score(model, X_test, y_test, cv=5)
			print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() *2))

			score[i][test_i] = scores.mean()
			if scores.mean()>best_mean:
				best_mean = scores.mean()
				best_test = test
				best_numFeat = i

	print('----- Finished Feature Selection ------')
	print('Best Mean: %05f\nBest Test: %s\nBest Number of Features: %d'%(best_mean, best_test, best_numFeat))
	print('Writing scores to .csv file')
	np.savetxt('../Data/kBestScores.csv',score,delimiter=",") 
	# print(score)





## Load data we will use
labels = np.load('../Data/big_labels.npy')
feat = np.load('../Data/big_features.npy')
feat = np.nan_to_num(feat)
print(labels.shape)
print(feat.shape)


# feat_x = feat.shape[0]
# print(feat_x)

# new_feat = feat.shape[0]/labels.shape[0]
# feat = feat.reshape(feat_x/new_feat,new_feat)
# print(feat.shape)


# np.nan_to_num(feat)
# X_train, X_test, y_train, y_test = train_test_split(
# feat, labels, test_size=0.2, random_state=0)

# y_train.ravel()
# model.fit(X_train,y_train)		

# scores = cross_val_score(model, X_test, y_test, cv=5)
# print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() *2))
univariateSelection(feat, labels)

## PCA Stuff
# z_scaler = StandardScaler()
# data = z_scaler.fit_transform(feat)

# pca = PCA(n_components=5)
# fit = pca.fit(data)

# print('PCA Explained Variance: %s' % fit.explained_variance_ratio_)
# plt.semilogy(fit.explained_variance_ratio_,'--o')
# plt.show()

# plt.semilogy(fit.explained_variance_ratio_.cumsum(),'--o')
# plt.show()


## Select k-best


