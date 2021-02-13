function features=readkaldifeatures(filename)

% READKALDIFEATURES Reads a set of features in Kaldi format
%
% features=readkaldifeatures(filename)
%
% Inputs:
% filename: Kaldi feature filename (.ARK extension) or script filename
% (.SCP extension)
%
% Output:
% features: Matlab structure with the following fields
% * features.utt{u}: name of utterance u
% * features.feature{u}: corresponding features (one row per feature and one
%   column per time frame)
%
% If you use this software in a publication, please cite
% Emmanuel Vincent and Shinji Watanabe, Kaldi to Matlab conversion tools, 
% http://kaldi-to-matlab.gforge.inria.fr/, 2014.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2014 Emmanuel Vincent (Inria) and Shinji Watanabe (MERL)
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strfind(filename,'scp')  % SCP file
    features=struct('utt',cell(1),'feature',cell(1));
    fid=fopen(filename,'r');
    utt_ind=1;
    prevfile='';
    while ~feof(fid),
        txt=fgetl(fid);
        b=strfind(txt,' ');
        e=strfind(txt,':');
        utt=txt(1:b-1);
        arkfile=txt(b+1:e-1);
        if ~strcmp(arkfile,prevfile),
            featuresfile=readkaldifeatures(arkfile);
        end
        uind=1;
        while ~strcmp(featuresfile.utt{uind},utt),
            uind=uind+1;
        end
        feature=featuresfile.feature{uind};
        features.utt{utt_ind}=utt;
        features.feature{utt_ind}=feature;
        utt_ind=utt_ind+1;
        prevfile=arkfile;
    end
else                        % ARK file
    %[~,~]=system(['copy-feats ark:' filename ' ark,t:' filename(1:end-3) 'txt']);
    fid=fopen([filename(1:end-3) 'txt'],'r');
    txt=fscanf(fid,'%c');
    fclose(fid);
    %system(['rm ' filename(1:end-3) 'txt']);
    features=struct('utt',cell(1),'feature',cell(1));
    b=strfind(txt,'[');
    e=[-1 strfind(txt,']')];
    for utt_ind=1:length(b),
        utt=txt(e(utt_ind)+2:b(utt_ind)-3);
        feature=eval(txt(b(utt_ind):e(utt_ind+1))).';
        features.utt{utt_ind}=utt;
        features.feature{utt_ind}=feature;
    end
end

return
