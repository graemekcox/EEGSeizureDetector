import numpy as np
from scipy.stats import skew, kurtosis
import scipy.io as spio

def fe_skewness(data):
	return np.array(skew(data)).reshape(-1,1)


def fe_kurtosis(data):
	return np.array(kurtosis(data)).reshape(-1,1)

def fe_variance(data):
	return np.array(np.var(data)).reshape(-1,1)


# fn = '/Users/graemecox/Documents/Capstone/Data/EEG_Data/Dog_1/Dog_1_ictal_segment_1.mat'

# mat = spio.loadmat(fn)
# data = mat['data']
# elec_data = data[1][:]

# # test = np.array([[1]])
# temp = fe_variance(elec_data)
# print(temp)
# print(temp.shape)
# print(np.append(test,temp,axis=0))

