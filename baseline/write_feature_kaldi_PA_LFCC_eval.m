clear; close all; clc;

% add required libraries to the path
addpath(genpath('LFCC'));
addpath(genpath('CQCC_v1.0'));
addpath(genpath('bosaris_toolkit'));
addpath(genpath('kaldi-to-matlab'));

% set here the experiment to run (access and feature type)
access_type = 'PA'; % LA for logical or PA for physical
feature_type = 'LFCC'; % LFCC or CQCC
data_type = 'eval'; % train or dev

mkdir('data\lfcc');
writefile = strcat('data\lfcc\', access_type, '_', feature_type, '_', data_type, '.txt');

pathToASVspoof2019Data = 'D:\Xu Li\anti-spoofing\ASVspoof2019\';

pathToDatabase = fullfile(pathToASVspoof2019Data, access_type);
if strcmp(data_type, 'train') % train 
    ProtocolFile = fullfile(pathToDatabase, horzcat('ASVspoof2019_', access_type, '_cm_protocols'), horzcat('ASVspoof2019.', access_type, '.cm.train.trn.txt'));
elseif strcmp(data_type, 'dev') % dev
    ProtocolFile = fullfile(pathToDatabase, horzcat('ASVspoof2019_', access_type, '_cm_protocols'), horzcat('ASVspoof2019.', access_type, '.cm.dev.trl.txt'));
elseif strcmp(data_type, 'eval') % dev
    ProtocolFile = fullfile(pathToDatabase, horzcat('ASVspoof2019_', access_type, '_cm_protocols'), horzcat('ASVspoof2019.', access_type, '.cm.eval.trl.txt'));
end

% read protocol
fileID = fopen(ProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s');
fclose(fileID);

% get file lists
filelist = protocol{2};

disp(writefile);
fid = fopen(writefile, 'w');

%% Feature extraction for data
disp('Extracting features for all data...');
% allFeatureCell = cell(size(filelist));
% allUttCell = cell(size(filelist));
for i=1:length(filelist)
    filePath = fullfile(pathToDatabase,['ASVspoof2019_' access_type strcat('_', data_type, '\flac')],[filelist{i} '.flac']);
    [x,fs] = audioread(filePath);
    if strcmp(feature_type,'LFCC')
        [stat,delta,double_delta] = extract_lfcc(x,fs,20,512,20);
	    Feature = [stat,delta,double_delta]';
    elseif strcmp(feature_type,'CQCC')
        Feature = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
    end
    Uttid = filelist{i};
    fprintf(fid, '%s  [\n ', Uttid);
    nfram = size(Feature, 2);
    for t = 1:nfram
        fprintf(fid, ' %.7g', Feature(:,t));
        fprintf(fid, ' \n ');
    end
    fprintf(fid, ' ]\n');
    if rem(i, 100) == 0
        disp(['Done ', num2str(i), ' utts.']);
    end
end
fclose(fid);
disp('Done!');

%% writekaldifeatures
%features = struct;
%features.utt = allUttCell;
%features.feature = allFeatureCell; 
%writekaldifeatures(features, strcat('feats\', access_type, '_', feature_type, '_', data_type, '.ark'));
%disp('finish stage 3')
