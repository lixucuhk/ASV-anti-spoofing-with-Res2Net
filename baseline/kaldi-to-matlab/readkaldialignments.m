function alignments=readkaldialignments(ali_filename,trans_filename,mdl_filename,tree_filename)

% READKALDIALIGNMENTS Reads a set of alignments in Kaldi format
%
% alignments=readkaldialignments(ali_filename,trans_filename)
%
% Inputs:
% ali_filename: Kaldi alignments filename (.GZ extension)
% trans_filename: Kaldi transcriptions filename (text format)
% mdl_filename: Kaldi acousticc model filename (optional, default:
% 'final.mdl' in the same directory as the alignments filename)
% tree_filename: Kaldi phonetic decision tree filename (optional, default:
% 'tree' in the same directory as the alignments filename)
%
% Output:
% alignments: Matlab structure with the following fields
% * alignments.utt{u}: name of utterance u
% * alignments.ali{u}.words: corresponding word sequence
% * alignments.ali{u}.seq: corresponding transition-id sequence
%
% Note: the corresponding model and phonetic decision tree must be located
% in the same directory as the alignment file and called final.mdl and
% tree, respectively
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

ali_dir=[fileparts(fileparts(ali_filename)) '/'];  % directory for model and tree
if nargin < 3,
    mdl_filename=[ali_dir 'final.mdl'];
    if nargin < 4,
        tree_filename=[ali_dir 'tree'];
    end
end

% Read transcriptions first
transcriptions=struct('utt',cell(1),'words',cell(1));
fid=fopen(trans_filename,'r');
utt_ind=1;
while ~feof(fid),
    txt=fgetl(fid);
    pos=strfind(txt,' ');
    transcriptions.utt{utt_ind}=txt(1:pos(1)-1);
    transcriptions.words{utt_ind}=txt(pos(1)+1:end);
    utt_ind=utt_ind+1;
end
fclose(fid);

% Now read alignments
[~,~]=system(['gunzip -c ' ali_filename ' |convert-ali ' mdl_filename ' ' mdl_filename ' ' tree_filename ' ark:- ark,t:' ali_filename(1:end-2) 'txt']);
alignments=struct('utt',cell(1),'ali',cell(1));
fid=fopen([ali_filename(1:end-2) 'txt'],'r');
utt_ind=1;
while ~feof(fid),
    txt=fgetl(fid);
    pos=strfind(txt,' ');    
    utt=txt(1:pos(1)-1);
    seq=uint16(eval(['[' txt(pos(1)+1:end) ']']));
    words='';
    % Find corresponding transcription (necessary because some utterances failed to be aligned)
    for u=1:length(transcriptions.utt),
        if strcmp(transcriptions.utt{u},utt),
            words=transcriptions.words{u};
        end
    end
    alignments.utt{utt_ind}=utt;
    alignments.ali{utt_ind}.words=words;
    alignments.ali{utt_ind}.seq=seq;
    utt_ind=utt_ind+1;
end
fclose(fid);
[~,~]=system(['rm ' ali_filename(1:end-2) 'txt']);

return