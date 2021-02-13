function writekaldinbests(nbests,nb_filename,dic_filename)

% WRITEKALDINBESTS Writes a set of N-best lists in Kaldi format
%
% writekaldinbests(nbests,nb_filename,dic_filename)
%
% Inputs:
% nbests: Set of N-best lists in Matlab format (see readkaldinbests for
% detailed format specification)
% nb_filename: Kaldi N-best lists filename (.GZ extension)
% dic_filename: Kaldi dictionary filename (typically called 'words.txt')
%
% Caution: the resulting file is not convertible into a factored lattice
% anymore, since the costs are arbitrarily assigned to the last word. It
% can still be scored, however.
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

lattices=struct('utt',cell(1),'lattice',cell(1));
utt_nbest_ind=1;
for utt_ind=1:length(nbests.utt),
    nbest=nbests.nbest{utt_ind};
    for n=1:length(nbest),
        utt=nbests.utt{utt_ind};
        utt_nbest=[utt '-' int2str(n)];
        words=nbest{n}.words;
        pos=[0 strfind(words,' ') length(words)+1];
        graph_cost=nbest{n}.graph_cost;
        acoustic_cost=nbest{n}.acoustic_cost;
        seq=nbest{n}.seq;
        lattice=cell(1);
        lattice{2}{3}.word='<eps>';
        lattice{2}{3}.graph_cost=[];
        lattice{2}{3}.acoustic_cost=[];
        lattice{2}{3}.seq=[];
        for state=3:length(pos),
            lattice{state}{state+1}.word=words(pos(state-1)+1:pos(state)-1);
            lattice{state}{state+1}.graph_cost=[];
            lattice{state}{state+1}.acoustic_cost=[];
            lattice{state}{state+1}.seq=[];
        end
        state=length(pos)+1;
        lattice{state}{1}.word='';
        lattice{state}{1}.graph_cost=graph_cost;
        lattice{state}{1}.acoustic_cost=acoustic_cost;
        lattice{state}{1}.seq=seq;
        lattices.utt{utt_nbest_ind}=utt_nbest;
        lattices.lattice{utt_nbest_ind}=lattice;
        utt_nbest_ind=utt_nbest_ind+1;
    end
end
writekaldilattices(lattices,nb_filename,dic_filename);

return