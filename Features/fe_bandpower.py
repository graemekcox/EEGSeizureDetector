import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import scipy
from pylab import *

freq_bands = {'Delta': (0,4),
			'Theta': (4,8),
			'Alpha': (8,12),
			'Beta': (12,30),
			'Gamma': (30,45)
			}

def fe_meanAmp(fn):


	mat = spio.loadmat(fn)

	Fs = mat['freq']
	data = np.array(mat['data'])

	num_elec = data.shape[0]

	features = []
	for i in range(num_elec):
		elec_data = data[i][:]



		amps=np.absolute(np.fft.rfft(elec_data))
		freqs = np.fft.rfftfreq(len(elec_data),1.0/Fs)

		# Take the mean of the fft amplitude for each EEG band
		temp_band = dict()
		for band in freq_bands:  
			#Find all indexs that belong to each EEG frequency band
		    i = np.where((freqs >= freq_bands[band][0]) & 
		                       (freqs <= freq_bands[band][1]))[0]
		    temp_band[band] = np.mean(amps[i])

		# print(L)
		# features = [][]
		values = [temp_band['Delta'],
			temp_band['Theta'],
			temp_band['Alpha'],
			temp_band['Beta'],
			temp_band['Gamma']]

		features.append(values)
		# print(np.array(values).shape)


	features = np.array(features)
	return features

	# for i in range(num_elec):
		# elec_data = data[i][:]

def fe_freqbandmean(data, Fs):
		amps=np.absolute(np.fft.rfft(data))
		freqs = np.fft.rfftfreq(len(data),1.0/Fs)

		# Take the mean of the fft amplitude for each EEG band
		temp_band = dict()
		for band in freq_bands:  
			#Find all indexs that belong to each EEG frequency band
		    i = np.where((freqs >= freq_bands[band][0]) & 
		                       (freqs <= freq_bands[band][1]))[0]
		    temp_band[band] = np.mean(amps[i])

		# print(L)
		# features = [][]
		values = [temp_band['Delta'],
			temp_band['Theta'],
			temp_band['Alpha'],
			temp_band['Beta'],
			temp_band['Gamma']]

		return values



# fn = '/Users/graemecox/Documents/Capstone/Data/EEG_Data/Dog_1/Dog_1_ictal_segment_1.mat'

# features = fe_meanAmp(fn)

# mat = spio.loadmat(fn)

# Fs = mat['freq']
# data = np.array(mat['data'])

# num_elec = data.shape[0]
# elec_data = data[0][:]



# eeg_band_fft = dict()

# fft_vals=np.absolute(np.fft.rfft(elec_data))
# fft_freq = np.fft.rfftfreq(len(elec_data),1.0/Fs)

# print(len(fft_freq))
# print(fft_freq[100])

# # Take the mean of the fft amplitude for each EEG band
# eeg_band_fft = dict()
# for band in freq_bands:  
#     freq_ix = np.where((fft_freq >= freq_bands[band][0]) & 
#                        (fft_freq <= freq_bands[band][1]))[0]
#     eeg_band_fft[band] = np.mean(fft_vals[freq_ix])

# # Plot the data (using pandas here cause it's easy)
# print(eeg_band_fft)
# print(eeg_band_fft['Theta'])

## Plotting spectrum
# fft1 = scipy.fft(elec_data)

# f = np.linspace(0,Fs,len(elec_data), endpoint=False)
# print(len(fft1))
# plt.figure(1)
# plt.plot(f, abs(fft1))
# plt.title('Mag')
# plt.xlabel('Frequency(Hz)')
# show()
