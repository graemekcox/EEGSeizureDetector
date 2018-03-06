## Wavelet feature extraction

import numpy as np
import pywt
import scipy.io as spio
from scipy.fftpack import fft
import matplotlib.pyplot as plt
# from scipy import signal


# for i in range(data.shape[0]):
# 	print i


# print(elec_data.shape)
# elec_data = elec_data.transpose()
# print(elec_data.shape)
# ## Wavelet decomp

# print(w.name)
# print(w.dec_len)
# a,d = pywt.dwt(elec_data,w,mode='constant')



# print(len(a))
# print(len(d))


# filt = np.convolve(a,elec_data)
# filt_D4 = np.convolve(d,elec_data)
# print(filt_D4.shape)
def mean(data):
	return sum(data)/float(len(data))

def quickFFT(data):
	Y = np.fft.fft(data)
	L = len(data)
	P2 = np.abs(Y/L)

	f_seiz = P2[0:L/2+1]
	f_seiz[1:L/2+1] = 2*f_seiz[1:L/2+1]
	return f_seiz


def fe_wavelet(fn):

	mat = spio.loadmat(fn)


	## FFT  
	sampling_freq = mat['freq']
	latency = mat['latency']

	data = np.array(mat['data'])


	num_elec = data.shape[0]
	print(num_elec)
	wname = 'db4';

	w = pywt.Wavelet(wname)


	features = []
	for i in range(num_elec):
		elec_data = data[i][:]


		f_seiz = quickFFT(elec_data)
		# print(L)
		# features = [][]
		values = [mean(f_seiz[49:75]),
			mean(f_seiz[75:100]),
			mean(f_seiz[124:150]),
			mean(f_seiz[149:175]),
			mean(f_seiz[174:200])]

		features.append(values)



	features = np.array(features)
	print(features.shape)
	return features

fn = '/Users/graemecox/Documents/Capstone/Data/EEG_Data/Dog_1/Dog_1_ictal_segment_1.mat'
fe_wavelet(fn)
