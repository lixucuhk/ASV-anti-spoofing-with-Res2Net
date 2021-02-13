function writekaldilattices(lattices,lat_filename,dic_filename)

% WRITEKALDILATTICES Writes a set of lattices in Kaldi format
%
% writekaldilattices(lattices,lat_filename,dic_filename)
%
% Inputs:
% lattices: set of lattices in Matlab format (see readkaldilattices for
% detailed format specification)
% lat_filename: Kaldi lattices filename (.GZ extension)
% dic_filename: Kaldi dictionary filename (typically called 'words.txt')
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

fid=fopen([lat_filename(1:end-2) 'txt'],'w');
for utt_ind=1:length(lattices.utt),
    utt=lattices.utt{utt_ind};
    lattice=lattices.lattice{utt_ind};
    fprintf(fid,'%s \n',utt);
    states1=[];                         % list state pairs
    for state1=1:length(lattice),
        if ~isempty(lattice{state1}),
            states1=[states1 state1];
        end
    end
    for state1=states1,
        states2=[];
        for state2=1:length(lattice{state1}),
            if ~isempty(lattice{state1}{state2}),
                states2=[states2 state2];
            end
        end
        for state2=states2,
            fprintf(fid,'%d',state1-2);
            word=lattice{state1}{state2}.word;
            if state2~=1,               % non-final state
                fprintf(fid,' %d %s',state2-2,word);
            end
            graph_cost=lattice{state1}{state2}.graph_cost;
            acoustic_cost=lattice{state1}{state2}.acoustic_cost;
            seq=lattice{state1}{state2}.seq;
            if ~isempty(graph_cost),    % costs defined
                fprintf(fid,' %.7g,%.7g,',graph_cost,acoustic_cost);
                if ~isempty(seq),       % path defined
                    fprintf(fid,'%d_',seq(1:end-1));
                    fprintf(fid,'%d',seq(end));
                end
            end
            fprintf(fid,' \n');
        end
    end
    fprintf(fid,' \n');
end
fclose(fid);
[~,~]=system(['sym2int.pl -f 3 ' dic_filename ' ' lat_filename(1:end-2) 'txt |lattice-copy --write-compact=true ark,t:- ark:- |gzip - > ' lat_filename]);
system(['rm ' lat_filename(1:end-2) 'txt']);

return