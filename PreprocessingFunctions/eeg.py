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

			self.latency = -mat['latency'] #only appears in ictal segments
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
	def __init__(self, id, age,weight, height,symptom, folder):
		self.id = id
		self.age = age
		self.weight = weight
		self.height = height
		self.folder = folder
		self.symptom = symptom

		self.eegList = []
		self.ictalSamples= []
		self.interictalSamples = []
		self.getEEGSamples()
		self.ictalIndex = 0
		self.normalSample = self.interictalSamples[0]
		

	def getEEGSamples(self):
		files = os.listdir(self.folder) # get all files from subfolder

		for file in glob.glob(self.folder+'*_ictal_*.mat'):
			eegTemp = EEG_Sample(file)
			self.eegList.append(eegTemp)
			self.ictalSamples.append(eegTemp)

		for file in glob.glob(self.folder+'*_interictal_*.mat'):
			eegTemp = EEG_Sample(file)
			self.eegList.append(eegTemp)
			self.interictalSamples.append(eegTemp)

	def setIctalIndex(self, index):
		self.ictalIndex = index

	def exportSeizureData(self):
		temp =self.ictalSamples[self.ictalIndex].data
		head = str(self.id)+','+str(self.age)+','+str(self.ictalIndex) +','+ str(self.weight)+','+ str(self.height)
		print('Wrote seizure to seizure.csv with following parameters: %s' % head)
		np.savetxt('../Data/seizure.csv',temp,header=head ,delimiter=",")  #Write out the file with rows being each electrode




# root = '/Users/graemecox/Documents/Capstone/Data/EEG_Data/'

# subfolders = os.listdir(root)
# subfolders.remove('.DS_Store')


# labels = []
# # num_feat = 10
# # features = np.empty((0,num_feat), float)
# features = np.array([])
# eegList = []

# fns = np.array(0,)
# # num_feat = 10

# print('------Starting to read from subfolders ------')
# for subfolder in subfolders:
# 	# print(root+subfolder)
# 	folder = root+subfolder
# 	print('Reading data from subfolder  %s' % folder)

# 	files = os.listdir(folder) #Get all files in subfolder

# 	#Find all files with certain keyword
# 	for file in glob.glob(folder+'/*_ictal_*.mat'):
# 		# ictal_clips.append(file)
# 		eegList.append(EEG_Sample(file))

# 	for file in glob.glob(folder+'/*_interictal_*.mat'):
# 		# interictal_clips.append(file)
# 		eegList.append(EEG_Sample(file))

# print('%d number of EEG Samples' % len(eegList))

# np.save('patients.npy')
# # feat, labels = readKaggleDataset(folder)

# # # # folder = '/Volumes/SeagateBackupPlusDrive/EEG_Data/SeizureDetectionData/'

# print(len(eegList))
# # ####### EXAMPLE #######
# folder = '/Users/graemecox/Documents/Capstone/Data/EEG_Data/Dog_1/'


# # # # First parameter is the patient ID
# dog1 = Patient(1317204,10,folder)

# # pat1.exportSeizureData() #Export next seizure in the list

# dog1.setIctalIndex(4) #Set new Index
# dog1.exportSeizureData() #Export next seizure in the list


# ### Normal segment


# mat = spio.loadmat('/Users/graemecox/Documents/Capstone/Data/EEG_Data/Dog_1/Dog_1_interictal_segment_70.mat')

# Fs = mat['freq']
# data = mat['data']
# head =  '1317204,10,Normal,30,50'
# np.savetxt('../Data/normal.csv',data ,header=head,delimiter=",")  #Write out the file with rows being each electrode


