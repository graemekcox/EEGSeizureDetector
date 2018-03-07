##kaggle reader
import numpy as np 
import scipy.io as spio
import os
import glob

fn = '/Users/graemecox/Documents/Capstone/Data/EEG_Data/Dog_1/Dog_1_ictal_segment_1.mat'


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

root = '/Users/graemecox/Documents/Capstone/Data/EEG_Data/'
# root = '/Volumes/SeagateBackupPlusDrive/EEG_Data/SeizureDetectionData/'

def getSubfolders(root):
	subfolders = os.listdir(root)
	subfolders.remove('.DS_Store')
	return subfolders
	# for folder in subfolders:
	# 	print(root+folder)
	# 	fns = os.listdir(root+folder)
	# 	print(len(fns))



def readKaggleDataset(root):
	subfolders = getSubfolders(root)

	interictal_clips = []
	ictal_clips = []
	test=[];
	test_clips = []


	for subfolder in subfolders:
		# print(root+subfolder)
		folder = root+subfolder
		print('Reading data from subfolder  %s' % folder)

		files = os.listdir(folder) #Get all files in subfolder

		#Find all files with certain keyword
		for file in glob.glob(folder+'/*_ictal_*.mat'):
			ictal_clips.append(file)

		for file in glob.glob(folder+'/*_interictal_*.mat'):
			interictal_clips.append(file)

		for file in glob.glob(folder+'/*_test_*.mat'):
			test_clips.append(file)


	print('Length of interictal clips is %d'% len(interictal_clips))
	print('Length of ictal clips is %d'% len(ictal_clips))
	print('Length of test clips is %d'% len(test_clips))

readKaggleDataset(root)