function nbests=readkaldinbests(nb_filename,dic_filename)

% READKALDINBESTS Reads a set of N-best lists in Kaldi format
%
% nbests=readkaldinbests(nb_filename,dic_filename)
%
% Inputs:
% nb_filename: Kaldi N-best lists filename (.GZ extension)
% dic_filename: Kaldi dictionary filename (typically called 'words.txt')
%
% Output:
% nbests: Matlab structure with the following fields
% * nbests.utt{u}: name of utterance u
% * nbests.nbest{u}{n}.words: n-th best word sequence
% * nbests.nbest{u}{n}.graph_cost: associated graph cost
% * nbests.nbest{u}{n}.acoustic_cost: associated acoustic cost
% * nbests.nbest{u}{n}.seq: associated transition-id sequence
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

lattices=readkaldilattices(nb_filename,dic_filename);
nbests=struct('utt',cell(1),'nbest',cell(1));
prev_utt='';
utt_ind=0;
for utt_nbest_ind=1:length(lattices.utt),
    utt_nbest=lattices.utt{utt_nbest_ind};
    pos=strfind(utt_nbest,'-');    
    utt=utt_nbest(1:pos(end)-1);
    if ~strcmp(utt,prev_utt),
        utt_ind=utt_ind+1;
        nbest=cell(1);
    end
    n=str2double(utt_nbest(pos(end)+1:end));
    lattice=lattices.lattice{utt_nbest_ind};
    words=[];
    graph_cost=0;
    acoustic_cost=0;
    seq=[];
    for state=3:length(lattice)-1,
        words=[words ' ' lattice{state}{state+1}.word];
        graph_cost=graph_cost+lattice{state}{state+1}.graph_cost;
        acoustic_cost=acoustic_cost+lattice{state}{state+1}.acoustic_cost;
        seq=[seq lattice{state}{state+1}.seq];
    end
    nbest{n}.words=words(2:end);
    nbest{n}.graph_cost=graph_cost;
    nbest{n}.acoustic_cost=acoustic_cost;
    nbest{n}.seq=seq;
    nbests.utt{utt_ind}=utt;
    nbests.nbest{utt_ind}=nbest;
    prev_utt=utt;
end

return