function [labels, features] = getReadSeizureData(root)

%     Get all interIcalClips in the folder
    interIctalClips = dir([root '*_interictal_*.mat']);
    ictalClips = dir([root '*_ictal_*.mat']);
    
    %Number of features being used
    numFeat = 5;
    features = zeros(1,numFeat);
%     labels = {};
    labels = zeros(1,1);
    
    disp('Reading in Interictal data');
    
    % Read in interictal clips
    for i=1:size(interIctalClips,1)

        file = [root interIctalClips(i).name];
        
        %Function takes in label and file name
%         [label,feat] = waveletFeatureExtractor(file, 'interictal');
        [label,feat] = waveletFeatureExtractor(file, 1);
        %%Add features and labels to arrays
        features(end+1:end+size(feat,1), 1:numFeat) = feat;
        labels = [labels; label'];
    end
    labels(1) = [];
    features(1,:) = []; %First row is filled with zeros
    
   
    %% Read in ictal clips
    disp('Reading in Ictal data')
    for i=1:size(ictalClips,1)

        file = [root ictalClips(i).name];
        
      %  Function takes in label and file name
%         [label,feat] = waveletFeatureExtractor(file, 'interictal');
        [label,feat] = waveletFeatureExtractor(file, -1);
        %Add features and labels to arrays
        features(end+1:end+size(feat,1), 1:numFeat) = feat;
        labels = [labels; label'];
    end
    testClips = dir([root '*_test_*.mat']);
%     for i=1:size(testClips,1)
% 
%         file = [root testClips(i).name];
%         segment = load(file);
%         data = segment.data;
% %       %  Function takes in label and file name
% % %         [label,feat] = waveletFeatureExtractor(file, 'interictal');
% %         [label,feat] = waveletFeatureExtractor(file, -1);
% %         %Add features and labels to arrays
% %         features(end+1:end+size(feat,1), 1:numFeat) = feat;
% %         labels = [labels; label'];
% %     end
end