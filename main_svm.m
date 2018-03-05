
%% Prep data for 2D-DWT
clear
close all
% root ='/Users/graemecox/Documents/Capstone/Data/EEG_Data/';
root = '/Volumes/SeagateBackupPlusDrive/EEG_Data/SeizureDetectionData/'


folders = dir(root);
folders(ismember({folders.name}, {'.','..','.DS_Store'})) = [];

numFeat = 5;
features = zeros(1,numFeat);
labels = zeros(1,1);

for j=1:length(folders)
    subfolder = fullfile(root,folders(j).name,'/');
    [temp_labels, temp_features] = getReadSeizureData(subfolder);
    fprintf('Finished finding all features for folder %s\n\n',folders(j).name)
    fprintf('Length of features: %d    Number of features: %d\n',size(temp_features,1),size(temp_features,2))
    fprintf('Length of labels: %d\n\n',size(temp_labels,1))
    features(end+1:end+size(temp_features,1), 1:numFeat) = temp_features;
    labels = [labels; temp_labels];
end

labels(1)
labels(1) = [];
labels(1)
features(1,:)  = []; % First row is filled with zeros

fileID = fopen('features.txt','w');
for i = 1:size(features,1)
    fprintf(fileID,'%f,%f,%f,%f,%f\n',features(i,1),features(i,2),features(i,3),features(i,4),features(i,5));
end
fclose(fileID);

fileID = fopen('labels.txt','w');
for i = 1:size(labels,1)
    fprintf(fileID,'%d\n',labels(i));
end
fclose(fileID);
fprintf('Finished writing to labels file\n')

% % Get all subfolders of our list
% patients = dir(root);
% patients(ismember({patients.name}, {'.','..','.DS_Store'})) = [];
% 
% for i=1:length(patients)
%     subfolder = fullfile(root,patients(i).name,'/');
%     [labels, features] = getReadSeizureData(subfolder);
% 
%     fprintf('Finished finding all features\n\n')
%     fprintf('Length of features: %d    Number of features: %d\n',size(features,1),size(features,2))
%     fprintf('Length of labels: %d\n\n',size(labels,1))
% end