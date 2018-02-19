
%% Prep data for 2D-DWT
clear
close all
root ='/Users/graemecox/Documents/Capstone/Data/EEG_Data/Dog_1/';
fn2 = 'Dog_5_interictal_segment_0001.mat';

[labels, features] = getReadSeizureData(root);

fprintf('Finished finding all features\n\n')

fprintf('Length of features: %d    Number of features: %d\n',size(features,1),size(features,2))
fprintf('Length of labels: %d\n\n',size(labels,1))



%% 2D-DWT example (probably will not work)

% features = waveletFeatureExtractor(file,'interictal');


%% Extracts features from Kaggle UPenn EEG Seizure detection data

