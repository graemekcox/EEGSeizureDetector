function [labels, features]  = waveletFeatureExtractor(file,label)
%% Extracts features from Kaggle UPenn EEG Seizure detection data
% Inputs are full-path to the eeg data.
    segment = load(file);

    %% label of 1 for interictal
    %% Label of -1 for ictal
    
    Fs = segment.freq;
    data = segment.data;

    num_elec = size(data,1);
    wname = 'db4';
    
    
    
    
    features = zeros(num_elec,5);
%     labels= {};
    labels = zeros(1,1);
    for i = 1:num_elec
        %Read in the electrode data
        data_elec = data(i,:);
        
        [LoD,HiD,LoR,HiR] = wfilters(wname);

        filt = conv(data_elec,LoD);
        filt_D4 = conv(filt,HiD);

    %     fft after filter
        Y = fft(filt_D4);
        L = length(filt_D4);
        P2 = abs(Y/L);
        f_seiz = P2(1:L/2+1);
        f_seiz(2:end-1) = 2*f_seiz(2:end-1);

        features(i,1) = mean(f_seiz(50:75));
        features(i,2) = mean(f_seiz(75:100));
        features(i,3) = mean(f_seiz(125:150));
        features(i,4) = mean(f_seiz(150:175));
        features(i,5) = mean(f_seiz(175:200));
%         labels{i } = label;
        labels(i) = label;
    end


end


