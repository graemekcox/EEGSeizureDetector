##kaggle reader
import numpy as np 
import scipy.io as spio

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


root = '/Volumes/SeagateBackupPlusDrive/EEG_Data/SeizureDetectionData/'

printMatData(fn)