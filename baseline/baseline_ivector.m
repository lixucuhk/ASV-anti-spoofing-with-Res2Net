%% get feature 
clear; close all; clc;

% 100_ivectors_la_dev_cqcc.txt
feat_dim  = 100;
access_type = 'LA'; % LA for logical or PA for physical
feature_type = 'MFCC'; % mfcc or cqcc
backend = 'guassian' % guassian or cosine

% add required libraries to the path
addpath(genpath('bosaris_toolkit'));
addpath(genpath('tDCF_v1'));
addpath(genpath('kaldi-to-matlab'));

pathToASVspoof2019Data = '/export/b14/jlai/ASVspoof2019-data/';

pathToDatabase = fullfile(pathToASVspoof2019Data, access_type);
trainProtocolFile = fullfile(pathToDatabase, horzcat('ASVspoof2019_', access_type, '_protocols'), horzcat('ASVspoof2019.', access_type, '.cm.train.trn.txt'));
devProtocolFile = fullfile(pathToDatabase, horzcat('ASVspoof2019_', access_type, '_protocols'), horzcat('ASVspoof2019.', access_type, '.cm.dev.trl.txt'));

% read train protocol
fileID = fopen(trainProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{2};
key = protocol{5};

% get indices of genuine and spoof files
bonafideIdx = find(strcmp(key,'bonafide'));
spoofIdx = find(strcmp(key,'spoof'));

kaldiTrain = fullfile('..','features','txt',strcat(num2str(feat_dim),'_ivectors_',lower(access_type),'_train_',lower(feature_type),'.txt'));
kaldiDev = fullfile('..','features','txt',strcat(num2str(feat_dim),'_ivectors_',lower(access_type),'_dev_',lower(feature_type),'.txt'));
kaldiTrain = char(kaldiTrain);
kaldiDev = char(kaldiDev);

% kaldi-to-matlab
feat_struct = readkaldifeatures(kaldiTrain);
utt = feat_struct.utt;
feat = feat_struct.feature;

disp('finish stage 1');

%% Feature extraction for training data
% extract features for GENUINE training data and store in cell array
disp('Extracting features for BONA FIDE training data...');
disp('Extracting features for SPOOF training data...');
genuineFeatureCell = cell(size(bonafideIdx));
spoofFeatureCell = cell(size(spoofIdx));
genuine_counter = 1;
spoof_counter = 1;
for i=1:length(utt)
    % get label
    index = find(strcmp(filelist, utt{i}));
    label = labels{index};
    if strcmp(label, 'genuine') % the utterance is genuine 
        genuineFeatureCell{genuine_counter} = feat{i};
        genuine_counter = genuine_counter + 1;
    else % the utterance is spoof 
        spoofFeatureCell{spoof_counter} = feat{i};
        spoof_counter = spoof_counter + 1;
    end 
end
disp('finish stage 2')

%% Generative Gaussian classifier from "Language Recognition in iVectors Space"
sum = 0; counter = 1;
for i=1:length(spoofFeatureCell)
    sum = sum + spoofFeatureCell{i};
    counter = counter + 1; 
end 
spoofFeatMean = sum/counter;
sum = 0; counter = 1;
for i=1:length(genuineFeatureCell)
    sum = sum + genuineFeatureCell{i};
    counter = counter + 1; 
end 
genuineFeatMean = sum/counter;
genuineTotalMat = [];
for i=1:length(genuineFeatureCell)
    genuineTotalMat = [genuineTotalMat; genuineFeatureCell{i}'];
end 
spoofTotalMat = [];
for i=1:length(spoofFeatureCell)
    spoofTotalMat = [spoofTotalMat; spoofFeatureCell{i}'];
end 
allMat = [spoofTotalMat;genuineTotalMat];
sharedCovMat = cov(allMat);
%sharedCovMat = bsxfun(@minus,spoofTotalMat,spoofFeatMean')'*bsxfun(@minus,genuineTotalMat,genuineFeatMean')/(size(spoofTotalMat,1)-1);
disp('finish stage 3')

%% Feature extraction and scoring of test data
% read development protocol
fileID = fopen(devProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{2};
attackType = protocol{4};
key = protocol{5};

% kaldi-to-matlab
feat_struct = readkaldifeatures(kaldiDev);
utt = feat_struct.utt;
feat = feat_struct.feature;

% process each development trial: feature extraction and scoring
scores_cm = zeros(size(filelist));
disp('Computing scores for development trials...');
parfor i=1:length(utt)
    x_feat = feat{i};
 
    if strcmp(backend, 'guassian')
        %score computation
        llk_genuine = -1/2*x_feat'*inv(sharedCovMat)*x_feat + x_feat'*inv(sharedCovMat)*genuineFeatMean - 1/2*genuineFeatMean'*inv(sharedCovMat)*genuineFeatMean;
        llk_spoof = -1/2*x_feat'*inv(sharedCovMat)*x_feat + x_feat'*inv(sharedCovMat)*spoofFeatMean - 1/2*spoofFeatMean'*inv(sharedCovMat)*spoofFeatMean;
        % compute log-likelihood ratio
        scores_cm(i) = llk_genuine - llk_spoof;
    else 
        % cosine similarity 
        cosine_genuine = dot(x_feat,genuineFeatMean)/norm(x_feat)/norm(genuineFeatMean);
        cosine_spoof   = dot(x_feat,spoofFeatMean)/norm(x_feat)/norm(spoofFeatMean);
        scores(i) = cosine_genuine - cosine_spoof;
    end
end
disp('finish stage 4')

%% save scores to disk
fid = fopen(fullfile('cm_scores',['scores_cm_' num2str(feat_dim) '_' ivector '_' access_type '_' feature_type '.txt']), 'w');
for i=1:length(scores_cm)
    fprintf(fid,'%s %s %s %.6f\n',filelist{i},attackType{i},key{i},scores_cm(i));
end
fclose(fid);

% compute performance
evaluate_tDCF_asvspoof19(fullfile('cm_scores', ['scores_cm_' num2str(feat_dim) '_' ivector '_' access_type '_' feature_type '.txt']), ...
    fullfile(pathToASVspoof2019Data, access_type, ['ASVspoof2019_' access_type '_dev_asv_scores_v1.txt']));
