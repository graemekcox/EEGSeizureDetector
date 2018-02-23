
%% Prep data for 2D-DWT
clear
close all
root ='/Users/graemecox/Documents/Capstone/Data/EEG_Data/Dog_1/';
fn2 = 'Dog_5_interictal_segment_0001.mat';

[labels, features] = getReadSeizureData(root);

fprintf('Finished finding all features\n\n')

fprintf('Length of features: %d    Number of features: %d\n',size(features,1),size(features,2))
fprintf('Length of labels: %d\n\n',size(labels,1))

%% Write features to a text file for python to read in
fileID = fopen('features.txt','w');ssz
fprintf(fileID,'Feature Extraction list\n')

for i=1:size(features,1)
    fprintf(fileID,'%f,%f,%f,%f,%f\n',features(i,1),features(i,2),features(i,3),features(i,4),features(i,5));
end
fclose(fileID);
fprintf('Finished writing to feature file\n')

fileID = fopen('labels.txt','w');
fprintf(fileID,'Labels List\n');
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