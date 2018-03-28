##kaggle reader
import sys
sys.path.insert(0, '/Users/graemecox/Documents/Capstone/Code/eegSvm/Features')
sys.path.insert(0, '/Users/graemecox/Documents/Capstone/Code/eegSvm/PreprocessingFunctions')

from fe_wavelet import *
from fe_bandpower import *
from fe_stats import *
from pyeeg import *
from eeg import *

import time


import numpy as np 
import scipy.io as spio
import os
import glob



# fn = '/Users/graemecox/Documents/Capstone/Data/EEG_Data/Dog_1/Dog_1_ictal_segment_1.mat'
num_feat = 818

def saveFiles(labels, features):
	np.save('Data/labels.npy', labels)
	print('Saved labels as Data/labels.npy')
	np.save('Data/features.npy', features)
	print('Saved features as Data/features.npy')

def printMatData(fn):
	mat = spio.loadmat(fn)

	print(mat['freq'])
	print(mat['latency'])
	print(np.array(mat['data']).shape)

	## Matlab structure fields:
	sampling_freq = mat['freq']
	latency = mat['latency']

	data = np.array(mat['data'])

	print(data.shape)

def getSubfolders(root):
	subfolders = os.listdir(root)
	subfolders.remove('.DS_Store')
	return subfolders
	# for folder in subfolders:
	# 	print(root+folder)
	# 	fns = os.listdir(root+folder)
	# 	print(len(fns))



def returnFeatures(fn):

	mat = spio.loadmat(fn)

	Fs = mat['freq']
	data = np.array(mat['data'])

	num_elec = data.shape[0]

	features = np.empty((0,num_feat))

	for i in range(num_elec):
		elec_data = data[i][:]

		temp_wav = fe_waveletdecomp(elec_data)
		# print(temp_feat.shape)
		temp_skew = fe_skewness(elec_data)
		temp_kurt = fe_kurtosis(elec_data)
		temp_var = fe_variance(elec_data)
		# # temp_feat = fe_kurtosis(elec_data)
		# # append all features vertically
		temp_features = np.append(temp_skew,temp_kurt, axis=1)
		temp_features = np.append(temp_features, temp_var,axis=1)
		temp_features = np.append(temp_features, temp_wav, axis=1)
		# print(temp_feat.shape)
		# temp_feat = np.concatenate((temp_skew, temp_kurt), axis=1)
		# print(temp_feat.shape)
		# temp_labels = np.array(eeg.label).reshape(-1,)

		# meanamp = fe_freqbandmean(elec_data,Fs)
		# w_db4 = fe_waveletdecomp(elec_data)
		# temp_features = fe_waveletdecomp(elec_data)
		# temp_features = fe_freqbandmean(elec_data,Fs)
		# temp_features = fe_spectralratio(elec_data,Fs)


		# append all features vertically
		# temp_features = np.concatenate((meanamp,w_db4), axis=1)
		
		#Append new features to next row in feature list
		features = np.append(features,temp_features,axis=0)

	return features

def returnLabels(feat, label):
	temp_size = feat.shape[0]
	labels = np.empty((temp_size,), int)
	labels[0:temp_size] = label #Ictal is -1
	return labels



def readKaggleDataset(root, save=0):
	subfolders = getSubfolders(root)

	interictal_clips = []
	ictal_clips = []
	test=[];
	test_clips = []

	labels = []
	# num_feat = 8

	features = np.empty((0,num_feat), float)

	print('----------------STARTING TO READ FROM SUBFOLDERS----------------')
	for subfolder in subfolders:
		# print(root+subfolder)
		folder = root+subfolder
		print('Reading data from subfolder  %s' % folder)

		files = os.listdir(folder) #Get all files in subfolder

		#Find all files with certain keyword
		for file in glob.glob(folder+'/*_ictal_*.mat'):
			ictal_clips.append(file)

			#Get features
			temp_feat = returnFeatures(file)
			temp_labels = returnLabels(temp_feat, 0)
			#Append labels to labels list
			# Append features and labels
			labels = np.append(labels, temp_labels, axis=0)
			features = np.append(features, temp_feat, axis=0)



		for file in glob.glob(folder+'/*_interictal_*.mat'):
			interictal_clips.append(file)

			#Get features
			temp_feat = returnFeatures(file)
			temp_labels = returnLabels(temp_feat,1)

			# Append features and labels
			labels = np.append(labels, temp_labels, axis=0)
			features = np.append(features, temp_feat, axis=0)

		for file in glob.glob(folder+'/*_test_*.mat'):
			test_clips.append(file)

	print('Length of interictal clips is %d'% len(interictal_clips))
	print('Length of ictal clips is %d'% len(ictal_clips))
	print('Length of test clips is %d'% len(test_clips))

	print('----------------Finished extracting features----------------')

	if save:
		saveFiles(labels, features)
		# np.save('Data/labels.npy', labels)
		# print('Saved labels as Data/labels.npy')
		# np.save('Data/features.npy', features)
		# print('Saved features as Data/features.npy')


	return features, labels, test_clips

def returnEEGSubjects(root):
	subfolders = getSubfolders(root)

	labels = []
	# num_feat = 10
	features = np.empty((0,num_feat), float)
	eegList = []

	fns = np.array(0,)
	# num_feat = 10
	features = np.empty((0,num_feat), float)

	print('------Starting to read from subfolders ------')
	for subfolder in subfolders:
		# print(root+subfolder)
		folder = root+subfolder
		print('Reading data from subfolder  %s' % folder)

		files = os.listdir(folder) #Get all files in subfolder

		#Find all files with certain keyword
		for file in glob.glob(folder+'/*_ictal_*.mat'):
			# ictal_clips.append(file)
			eegList.append(EEG_Sample(file))

		for file in glob.glob(folder+'/*_interictal_*.mat'):
			# interictal_clips.append(file)
			eegList.append(EEG_Sample(file))

	print('%d number of EEG Samples' % len(eegList))

	np.save('/Users/graemecox/Documents/Capstone/Code/eegSvm/Data/eeg_samples',eegList)
	return eegList

def getFeaturesForEEGSample(eegSamples, save=0):
	
	labels = np.array([])
	# num_feat = 8

	# features = np.empty((0,num_feat))
	# features = np.array([])
	features = np.empty((0,19))
	temp_label = np.empty((1,),int)

	print('----- Starting to read from samples -----')
	eeg_count = 1
	start = time.time()

	print('There are %d number of EEG samples'%len(eegSamples))
	for eeg in eegSamples:
		eeg_count = eeg_count + 1

		if (eeg_count%100 == 0):
			elapsed_time = time.time() - start
			print('Processing EEG #%d'%eeg_count)
			print('Elapsed Time: %04f' % elapsed_time)


		temp_data = eeg.data

		num_elec = temp_data.shape[0]

		for i in range(num_elec):
			# temp_feat = np.array([])
			elec_data = temp_data[i][:]

			# print('PFD Size: (%d,%d)' % (temp_pfd.shape[0],temp_pfd.shape[1]))
			# temp_feat = np.append(temp_feat,pfd(elec_data),axis=1)

			# temp_feat = pfd(elec_data)



			temp_bin =  np.array(bin_power(elec_data, [0.5,4,7,12,30] , eeg.Fs[0]))
			temp_bin =temp_bin.reshape(1,8) 

			temp_feat = np.append(fe_waveletdecomp(elec_data), temp_bin,axis=1)

			temp_pfd = np.array(pfd(elec_data)).reshape(-1,1)
			temp_feat = np.append(temp_feat,temp_pfd,axis=1)

			temp_hurst = np.array(hurst).reshape(-1,1)
			temp_feat = np.append(temp_feat,temp_hurst,axis=1)

			## Size (400,1)
			# temp_es = embed_seq(elec_data,4,1)
			# print(temp_es.shape)
			# temp_feat = np.append(temp_feat, embed_seq(elec_data,4,1))

			# temp_bin =  bin_power(elec_data, [0.5,4,7,12,30] , eeg.Fs[0])
			# temp_feat = np.append(temp_feat, temp_bin,axis=1)

			# print(temp_feat)
			##Size 399
			temp_diff=  first_order_diff(elec_data)
			# print(len(temp_diff))
			# temp_feat = np.append(temp_feat, temp_diff)
			temp_hfd = np.array(hfd(elec_data,100)).reshape(-1,1)
			temp_feat = np.append(temp_feat, temp_hfd,axis=1) #kMax set to 100 for now, not sure how high this is

			temp_hj = np.array(hjorth(elec_data, temp_diff)).reshape(-1,2)
			temp_feat = np.append(temp_feat, temp_hj,axis=1)

			temp_se = np.array(spectral_entropy(elec_data, [0.5,4,7,12,30] , eeg.Fs)).reshape(-1,1)
			temp_feat = np.append(temp_feat, temp_se,axis=1)
			# temp_feat = fe_waveletdecomp(elec_data)
			# print(temp_feat.shape)
			# temp_skew = fe_skewness(elec_data)
			# temp_kurt = fe_kurtosis(elec_data)
			# temp_var = fe_variance(elec_data)
			# # # temp_feat = fe_kurtosis(elec_data)
			# # # append all features vertically
			# temp_feat = np.append(temp_skew,temp_kurt, axis=1)
			# temp_feat = np.append(temp_feat, temp_var,axis=1)
			# temp_feat = np.append(temp_feat, temp_wav, axis=1)
			# print(temp_feat.shape)
			# temp_feat = np.concatenate((temp_skew, temp_kurt), axis=1)
			# print(temp_feat.shape)
			temp_labels = np.array(eeg.label).reshape(-1,)


			# print(temp_feat.shape)
			labels = np.append(labels, temp_labels, axis=0)
			# print(features.shape)

			features = np.append(features, temp_feat, axis=0)
			# print(features.shape)

	if save:
		saveFiles(labels, features)
	print('------ Finished reading through all samples -----')

	return features, labels





folder = '/Users/graemecox/Documents/Capstone/Data/EEG_Data/'

# # feat, labels = readKaggleDataset(folder)

# # # # folder = '/Volumes/SeagateBackupPlusDrive/EEG_Data/SeizureDetectionData/'
eegList = returnEEGSubjects(folder)
# # # fn = '/Users/graemecox/Documents/Capstone/Code/eegSvm/Data/eeg_samples_Dog_1_3.npy'
# # # eegList = np.load(fn)

# # smallList = eegList[0:5]
smallList = eegList[0:len(eegList)/3]

feat, labels = getFeaturesForEEGSample(smallList)
print(feat.shape)
print(labels.shape)
np.save('/Users/graemecox/Documents/Capstone/Code/eegSvm/Data/big_labels.npy', labels)
print('Saved labels as Data/big_labels.npy')
np.save('/Users/graemecox/Documents/Capstone/Code/eegSvm/Data/big_features.npy',feat)
print('Saved features as Data/big_features.npy')





