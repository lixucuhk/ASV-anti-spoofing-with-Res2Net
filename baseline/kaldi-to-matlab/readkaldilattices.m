function lattices=readkaldilattices(lat_filename,dic_filename)

% READKALDILATTICES Reads a set of lattices in Kaldi format
%
% lattices=readkaldilattices(lat_filename,dic_filename)
%
% Inputs:
% lat_filename: Kaldi lattices filename (.GZ extension)
% dic_filename: Kaldi dictionary filename (typically called 'words.txt')
%
% Output:
% lattices: Matlab structure with the following fields
% * lattices.utt{u}: name of utterance u
% * lattices.lattice{u}{n1}{n2}.word: word associated with transition from
%   node n1 to node n2 of the lattice
% * lattices.lattice{u}{n1}{n2}.graph_cost: associated graph cost
% * lattices.lattice{u}{n1}{n2}.acoustic_cost: associated acoustic cost
% * lattices.lattice{u}{n1}{n2}.seq: associated transition-id sequence
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

[~,~]=system(['gunzip -c ' lat_filename ' |lattice-copy --write-compact=true ark:- ark,t:- |int2sym.pl -f 3 ' dic_filename ' > ' lat_filename(1:end-2) 'txt']);
fid=fopen([lat_filename(1:end-2) 'txt'],'r');
lattices=struct('utt',cell(1),'lattice',cell(1));
utt_ind=1;
while ~feof(fid),
    lattice=cell(1);
    txt=fgetl(fid);
    utt=txt(1:end-1);
    txt=fgetl(fid);
    while ~isempty(txt),
        pos=strfind(txt,' ');
        state1=eval(txt(1:pos(1)-1))+2; % adding 2 to all states (1 = final state)
        if length(pos) > 2,     % non-final state
            state2=eval(txt(pos(1)+1:pos(2)-1))+2;
            word=txt(pos(2)+1:pos(3)-1);
            txt=txt(pos(3)+1:end);
        else                    % final state
            state2=1;
            word='';
            txt=txt(pos(1)+1:end);
        end
        pos=strfind(txt,',');
        if ~isempty(pos),       % costs defined
            graph_cost=eval(txt(1:pos(1)-1));
            acoustic_cost=eval(txt(pos(1)+1:pos(2)-1));
            txt=txt(pos(2)+1:end);
            if ~isempty(txt),   % path defined
                seq=uint16(eval(['[' strrep(txt,'_',' ') ']']));
            else                % path undefined
                seq=[];
            end
        else                    % costs undefined
            graph_cost=[];
            acoustic_cost=[];
            seq=[];
        end
        lattice{state1}{state2}.word=word;
        lattice{state1}{state2}.graph_cost=graph_cost;
        lattice{state1}{state2}.acoustic_cost=acoustic_cost;
        lattice{state1}{state2}.seq=seq;
        txt=fgetl(fid);
    end
    lattices.utt{utt_ind}=utt;
    lattices.lattice{utt_ind}=lattice;
    utt_ind=utt_ind+1;
end
fclose(fid);
system(['rm ' lat_filename(1:end-2) 'txt']);

return