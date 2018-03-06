function [labels, features,test_data] = getReadSeizureData(root)

%     Get all interIcalClips in the folder
    interIctalClips = dir([root '*_interictal_*.mat']);
    ictalClips = dir([root '*_ictal_*.mat']);
    testClips = dir([root '*_test_*.mat']);
    
    %Number of features being used
    numFeat = 5;

    features = [];
    labels = [];

    disp('Reading in Interictal data');
    
    % Read in interictal clips
    for i=1:size(interIctalClips,1)

        file = [root interIctalClips(i).name];
        
        %Function takes in label and file name

        [feat] = waveletFeatureExtractor(file);
        
        %% Append features and labels to array
        features = [features;feat];
        temp_label = zeros(1,size(feat,1));
        temp_label(:) = 1; %For non-seizure segment
        labels = [labels;temp_label'];        

    end
    
    %% Read in ictal clips
    for i=1:size(ictalClips,1)

        file = [root ictalClips(i).name];
        
        %Function takes in label and file name
        [feat] = waveletFeatureExtractor(file);
        
        %%Add features and labels to arrays
        features = [features;feat];
        temp_label = zeros(1,size(feat,1));
        temp_label(:) = -1; %For seizure segments

        labels = [labels;temp_label'];
    end
    %% Read in test clips
    
    disp('Reading in Test data');
    
    test_data = [];
    for i=1:size(testClips,1)
        file = [root testClips(i).name];
        
        %Function takes in label and file name
        [feat] = waveletFeatureExtractor(file);
        
        %%Add features and labels to arrays
        test_data = [test_data;feat];
    end

end