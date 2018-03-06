
%% Prep data for 2D-DWT
clear
close all
% root ='/Users/graemecox/Documents/Capstone/Data/EEG_Data/';
root = '/Volumes/SeagateBackupPlusDrive/EEG_Data/SeizureDetectionData/'

%% Initial script parameters
WRITE_TXT = 1; %either write to text, or csv
% 0 for csv, 1 for text


folders = dir(root);
folders(ismember({folders.name}, {'.','..','.DS_Store'})) = [];

numFeat = 5;
% features = zeros(1,numFeat);
% labels = zeros(1,1);
features = [];
labels = [];

test_data = [];
% test_data = zeros(1,numFeat);%Hold features and labels


fprintf('Beginning to extract features!\n\n')
tic

for j=1:length(folders)
    subfolder = fullfile(root,folders(j).name,'/');
    %% Get all labels and features of subfolders, and return filesnames for test data
    [temp_labels, temp_features, temp_test] = getReadSeizureData(subfolder);
    
    fprintf('Finished finding all features for folder %s\n\n',folders(j).name)

    %% Append data to existing feature and label list.
%     features(end+1:end+size(temp_features,1), 1:numFeat) = temp_features;
    features = [features; temp_features];
    labels = [labels; temp_labels];
    
    %%Append test data to test set
    test_data = [test_data;temp_test];
end

% features(1,:)  = []; % First row is filled with zeros
% labels(1,:) = [];
% test_data(1,:) = [];

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
    
    csvwrite('testData.csv',test_data);
    fprintf('\nTest features have been written to testData.csv\n')
end


toc
fprintf('ALL DONE!\n')