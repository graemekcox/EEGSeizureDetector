function [labels, features] = getReadSeizureData(root)

    %Get all interIcalClips in the folder
    interIctalClips = dir([root '*_interictal_*.mat']);
    
    %Number of features being used
    numFeat = 5;
    features = zeros(1,numFeat);
    labels = {};
    
    disp('Reading in Interictal data');
    
    
    index = 1;
    for i=1:size(interIctalClips,1)

        file = [root interIctalClips(i).name];
        
        %Function takes in label and file name
        [label,feat] = waveletFeatureExtractor(file, 'interictal');
        
        %%Add features and labels to arrays
        features(end+1:end+size(feat,1), 1:numFeat) = feat;
        labels = [labels; label'];
    end
    features(1,:) = []; %First row is filled with zeros
    
end