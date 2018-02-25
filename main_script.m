clc;
clear all;
close all;

% Load in Data

path = '/Users/graemecox/Documents/Capstone/Data/EEG_Data/Dog_1';
file = 'Dog_1_interictal_segment_1.mat';


fileID = load(fullfile(path,file));

wname = 'db4';
num_elec = size(fileID.data,1);


Fs = 400; % fileID.interictal_segment_1.sampling_frequency;
Ts = .1;  % STFT size
overlap = .25; % percentage of overlap

nfft = Ts*Fs;  
noverlap = overlap*nfft;
win = hamming(nfft);

feat1 = zeros(num_elec,1);
feat2 = zeros(num_elec,1);
feat3 = zeros(num_elec,1);
feat4 = zeros(num_elec,1);
feat5 = zeros(num_elec,1);
% feat6 = zeros(num_elec,1);


% if isempty(strfind(file,


for i = 1:num_elec
    %% Read in data for each electrode
    data = fileID.data(i,:);

    %% Wavelet analysis

    [LoD,HiD,LoR,HiR] = wfilters(wname);

    filt = conv(data,LoD);
    filt_D4 = conv(filt,HiD);

%     fft after filter
    Y = fft(filt_D4);
    L = length(filt_D4);
    P2 = abs(Y/L);
    f_seiz = P2(1:L/2+1);
    f_seiz(2:end-1) = 2*f_seiz(2:end-1);

    feat1(i,1) = mean(f_seiz(50:75));
    feat2(i,1) = mean(f_seiz(75:100));
    feat3(i,1) = mean(f_seiz(125:150));
    feat4(i,1) = mean(f_seiz(150:175));
    feat5(i,1) = mean(f_seiz(175:200));
    label{i} = 'interictal';
%     figure()
%     subplot(2,1,1)
%     plot(filt_D4)
%     title('wavelet filt')
%     subplot(2,1,2)
%     plot(data)
%     title('raw')
%     
%     figure()
%     subplot(2,1,1)
%     plot(f_seiz)
%     title('fft with filter')
%     subplot(2,1,2)
%     plot(fft_seiz)
%     title('fft no filter')
    
end