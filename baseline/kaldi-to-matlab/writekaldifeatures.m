function writekaldifeatures(features,filename)

% WRITEKALDIFEATURES Writes a set of features in Kaldi format
%
% writekaldifeatures(features,filename)
%
% Inputs:
% features: set of features in Matlab format (see readkaldifeatures for
% detailed format specification)
% filename: Kaldi feature filename (.ARK extension)
%
% Note: a .SCP file containing the location of the output .ARK file is also
% created
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

fid=fopen([filename(1:end-3) 'txt'],'w');
for utt_ind=1:length(features.utt),
    utt=features.utt{utt_ind};
    feature=features.feature{utt_ind};
    fprintf(fid,'%s  [\n ', utt);
    nfram=size(feature,2);
    for t=1:nfram,
        fprintf(fid,' %.7g', feature(:,t));
        fprintf(fid,' \n ');
    end
    fprintf(fid,' ]\n');
end
fclose(fid);
%[~,~]=system(['copy-feats --compress=true ark,t:' filename(1:end-3) 'txt ark,scp:' filename ',' filename(1:end-3) 'scp']);
%system(['rm ' filename(1:end-3) 'txt']);

return
