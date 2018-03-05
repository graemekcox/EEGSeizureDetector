
%% Prep data for 2D-DWT
clear
close all
root ='/Users/graemecox/Documents/Capstone/Data/EEG_Data/';
% root = '/Volumes/SeagateBackupPlusDrive/EEG_Data/SeizureDetectionData/'

%% Initial script parameters
WRITE_TXT = 0; %either write to text, or csv
% 0 for csv, 1 for text


folders = dir(root);
folders(ismember({folders.name}, {'.','..','.DS_Store'})) = [];

numFeat = 5;
features = zeros(1,numFeat);
labels = zeros(1,1);

fprintf('Beginning to extract features!\n\n')
tic

for j=1:length(folders)
    subfolder = fullfile(root,folders(j).name,'/');
    %% Get all labels and features of subfolders, and return filesnames for test data
    [temp_labels, temp_features, temp_test] = getReadSeizureData(subfolder);
    
    fprintf('Finished finding all features for folder %s\n\n',folders(j).name)
%     fprintf('Length of features: %d    Number of features: %d\n',size(temp_features,1),size(temp_features,2))
%     fprintf('Length of labels: %d\n\n',size(temp_labels,1))
%     
    %% Append data to existing feature and label list.
    features(end+1:end+size(temp_features,1), 1:numFeat) = temp_features;
    labels = [labels; temp_labels];
end

features(1,:)  = []; % First row is filled with zeros
labels(1,:) = [];

fprintf('\n\n----------Feature Extraction complete----------\n\n')

%% Double-check that labels and features match up
if (size(labels,1) ~= size(features,1))
    fprintf('ERROR: The length of labels and features are not equal! How could this happen????')
end

fprintf('Total length of labels: %d\nTotal length of features: %d\n',size(labels,1), size(features,1))

if (WRITE_TXT)
    %Write features file
    fileID = fopen('features.txt','w');
    for i = 1:size(features,1)
        fprintf(fileID,'%f,%f,%f,%f,%f\n',features(i,1),features(i,2),features(i,3),features(i,4),features(i,5));
    end
    fclose(fileID);
    %Write labels file
    fileID = fopen('labels.txt','w');
    for i = 1:size(labels,1)
        fprintf(fileID,'%d\n',labels(i));
    end
    fclose(fileID);
    fprintf('\nFinished writing to labels file\n')
else
    data = zeros(length(labels), size(features,2)+1);
    data(:,1) = labels;
    data(:,2:end) = features(:,:);
    csvwrite('features.csv',data);
    fprintf('\nFeatures and labels written to features.csv\n')
end
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
toc
fprintf('ALL DONE!\n')