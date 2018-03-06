clear
clc
close all


root = '/users/graemecox/Downloads/SeizureDetectionData';

folders = dir(root);
folders = folders(~ismember({folders.name},{'.','..','.DS_Store'}));

% 
% %%Go through all folders
% 
% for i=1:length(folders)
%     
%     
% end

subfolder = folders(1);

% newfolder = fullfile(root,subfolder.name);
newPath = fullfile(root,subfolder.name);
files = dir(newPath); %%Get all files inside the subfolder
files = files(~ismember({files.name},{'.','..','.DS_Store'}));



% for i=1:length(files);
%     file = fullfile(newPath,files(i));
%     data = load(file);
% end

file = [newPath,'/',files(1).name];
file1 = fullfile(newPath,files(1).name);

% file = '/Users/graemecox/Downloads/SeizureDetectionData/Patient_1/Patient_1_interictal_segment_53.mat';

eeg = load(file);

data=eeg.data;

%% Power spectral analysis
N = length(data);
n = 0:N-1;
Fs = eeg.freq;
index = 4;

windowsize = 128;
nfft = windowsize;
noverlap = windowsize-1;
spectrogram(data(index,:),windowsize,noverlap,nfft,Fs)
title('STFT of Seizure Data')
% [S, F, T]=spectrogram(data(1,:),windowsize,noverlap,nfft,Fs)
% % 128 for hamming window of length 128
% % 120 samples of overlap between sections
% % 128 
% imagesc(T,F,log10(abs(S)));
% set(gca,'YDir','Normal')
% ylabel('Time (s)')
% xlabel('Freq (Hz)')
% title('STFT')


file = '/Users/graemecox/Downloads/SeizureDetectionData/Patient_1/Patient_1_interictal_segment_53.mat';

eeg = load(file);

data=eeg.data;

%% Power spectral analysis

figure
spectrogram(data(index,:),128,127,128,Fs)
title('STFT Of Interictal Segment')