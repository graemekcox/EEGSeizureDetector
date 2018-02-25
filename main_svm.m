
%% Prep data for 2D-DWT
clear
close all

root = '/Users/graemecox/Documents/Capstone/Data/EEG_Data/';
folders = dir(root);
folders(ismember({folders.name},{'.','..','.DS_Store'})) = [];

numFeat = 5;
features = zeros(1,numFeat);
labels = zeros(1,1);
for j=1:length(folders)
    subfolder = fullfile(root, folders(j).name,'/')

    
    [temp_labels, temp_features] = getReadSeizureData(subfolder);

    fprintf('Finished finding all features\n\n')

    fprintf('Length of features: %d    Number of features: %d\n',size(temp_features,1),size(temp_features,2))
    fprintf('Length of labels: %d\n\n',size(temp_labels,1))

    
    features(end+1:end+size(temp_features,1), 1:numFeat) = temp_features;
    labels = [labels; temp_labels];
    

end

labels(1) = [];
features(1,:) = []; %First row is filled with zeros


%% Write features to a text file for python to read in
fileID = fopen('features.txt','w');
% fprintf(fileID,'Feature Extraction list\n')

for i=1:size(features,1)
    fprintf(fileID,'%f,%f,%f,%f,%f\n',features(i,1),features(i,2),features(i,3),features(i,4),features(i,5));
end
fclose(fileID);
fprintf('Finished writing to feature file\n')

fileID = fopen('labels.txt','w');
% fprintf(fileID,'Labels List\n');
for i=1:size(labels,1)
   fprintf(fileID,'%d\n',labels(i));
end
fclose(fileID);
fprintf('Finished writing to labels file\n')

    
%% Extracts features from Kaggle UPenn EEG Seizure detection data
% 
% % cdata = [grnpts;redpts];
% % grp = ones(200,1);
% % grp(101:200) = -1;
% % length(labels(:,1))
% % %Cross-validation
% c = cvpartition(size(labels,1),'KFold',10);
% 
% sigma = optimizableVariable('sigma',[1e-5,1e5],'Transform','log');
% box = optimizableVariable('box',[1e-5,1e5],'Transform','log');
% minfn = @(z)kfoldLoss(fitcsvm(features,labels,'CVPartition',c,...
%     'KernelFunction','rbf','BoxConstraint',z.box,...
%     'KernelScale',z.sigma));
% %%Optimize classifier
% results = bayesopt(minfn,[sigma,box],'IsObjectiveDeterministic',true,...
%     'AcquisitionFunctionName','expected-improvement-plus')
% 
% %% Use results for classification
% z(1) = results.XAtMinObjective.sigma;
% z(2) = results.XAtMinObjective.box;
% SVMModel = fitcsvm(features,labels,'KernelFunction','rbf',...
%     'KernelScale',z(1),'BoxConstraint',z(2));
% d= 0.02;
% [x1Grid,x2Grid] = meshgrid(min(features(:,1)):d:max(features(:,1)),...
%     min(features(:,2)):d:max(features(:,2)));
% xGrid = [x1Grid(:),x2Grid(:)];
% [~,scores] = predict(SVMModel,xGrid);