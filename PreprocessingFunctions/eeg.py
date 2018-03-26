import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
import os
import glob


class EEG_Sample:

	def __init__(self, path):
		self.fn = path

		mat = spio.loadmat(path)
		self.Fs = mat['freq']
		self.data = mat['data']
		self.numElec = self.data.shape[0]
		tempFolder = np.array(path.split('/'))
		self.subject = tempFolder[len(tempFolder) - 2] #Specifies which subject the data is from

		# self.label = 
		if (-1 != path.find('_ictal_')):
			self.label = 0

			self.latency = -mat['latency']. #only appears in ictal segments
			#time between seizure onset and first data point in data segment
		elif (-1 != path.find('_interictal_')):
			self.label = 1

			self.latency = -1
		else:
		 	self.label = -1 # for test data. 

			self.latency = -1

		#find 
	def setData(self, new_data):
		self.data = new_data

	def plotTimeSeries(self, elecNum):
		if (elecNum > self.numElec):
			print('Electrode number not valid for this dataset')
			return

		elec_data = self.data[elecNum-1][:]
		t = np.arange(len(elec_data)) * (1/self.Fs)

		plt.plot(t,elec_data)
		plt.xlabel('Time (s)')
		plt.ylabel('Voltage (mV)')
		plt.title('Time Series of Electrode #%d' % elecNum)
		plt.show()


	def plotFourierSeries(self, elecNum):
		if (elecNum > self.numElec):
			print('Electrode number not valid for this dataset')
			return
		elec_data = self.data[elecNum-1]
		mag = np.absolute(np.fft.rfft(elec_data))

		freqs = np.fft.rfftfreq(len(elec_data), 1.0/self.Fs)

		plt.plot(freqs,mag)
		plt.xlabel('Frequency (Hz)')
		plt.ylabel('Magnitude')
		plt.title('Fourier Series of Electrode #%d' % elecNum)
		plt.show()


	# def addNoise(data):
class Patient:
	def __init__(self, id, age, folder):
		self.id = id
		self.age = age
		self.folder = folder

		self.eegList = []
		self.ictalSamples= []
		self.getEEGSamples()
		

	def getEEGSamples(self):
		files = os.listdir(self.folder) # get all files from subfolder

		for file in glob.glob(self.folder+'*_ictal_*.mat'):
			eegTemp = EEG_Sample(file)
			self.eegList.append(eegTemp)
			self.ictalSamples.append(eegTemp)

		for file in glob.glob(self.folder+'*_interIctal_*.mat'):
			eegTemp = EEG_Sample(file)
			self.eegList.append(eegTemp)





# fn = '/Users/graemecox/Documents/Capstone/Data/EEG_Data/Dog_3/Dog_3_ictal_segment_17.mat'
# test = EEG_Sample(fn)

# # print(test.fn)

# mat = spio.loadmat(fn)
# print(mat['latency'])	


# folder = '/Users/graemecox/Documents/Capstone/Data/EEG_Data/Dog_3/'


# # for file in glob.glob(folder):
# # 	print(file)
# dog1 = Patient(1317204,22,folder)
# print(len(dog1.eegList))

# ictalPeriods = dog1.ictalSamples
# print(ictalPeriods[0].latency)


# print(test.subject)
# test.plotFourierSeries(60)
# test.plotTimeSeries(17)
