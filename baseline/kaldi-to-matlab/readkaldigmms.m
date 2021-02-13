function models=readkaldigmms(filename)

% READKALDIGMMS Reads a set of GMM-HMM models with diagonal covariance in
% Kaldi format
%
% models=readkaldigmms(filename)
%
% Input:
% filename: Kaldi GMM-HMM filename (.MDL extension)
%
% Output:
% models: Matlab structure with the following fields
% * models.TransitionModel.Topology{p}{h}.PdfClass: pdf-id for HMM-state h
%   of phone p
% * models.TransitionModel.Topology{p}{h}.Transition: transition
%   probabilities for HMM-state h of phone p
% * models.TransitionModel.Triples: matrix listing the phone (1st column),
%   the HMM-state (2nd column), and the pdf-id (3rd column) corresponding
%   to each transition-state (one per line)
% * models.TransitionModel.LogProbs: log-probability of self-transition for
%   each transition-id (one per row)
% * models.TransitionModel.State2id: transition-id corresponding to each
%   transition-state (one per row)
% * models.TransitionModel.Id2state: transition-state corresponding to each
%   transition-id (one per row)
% * models.Pdfs{i}.Gconsts: vector of log-normalization factors (one column
%   per Gaussian component)
% * models.Pdfs{i}.Weights: vector of weights (one column per Gaussian
%   component)
% * models.Pdfs{i}.Means_invvars: product of the means and the inverse
%   variances (one row per feature and one column per Gaussian component)
% * models.Pdfs{i}.Inv_vars: inverse variances (one row per feature and one
%   column per Gaussian component)
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

[~,~]=system(['gmm-copy --binary=false ' filename ' ' filename(1:end-3) 'txt']);
fid=fopen([filename(1:end-3) 'txt'],'r');
fgetl(fid); %<TransitionModel>
fgetl(fid); %<Topology>
txt=fgetl(fid);
Topology=cell(1);
while strcmp(txt,'<TopologyEntry> '),
    fgetl(fid); %<ForPhones>
    txt=fgetl(fid);
    forphones=eval(['[' txt ']']);
    fgetl(fid); %</ForPhones>
    txt=fgetl(fid);
    phonemodel=cell(1);
    state=1;    % adding +1 to State and PdfClass
    while strfind(txt,'<PdfClass> '),  % nonempty state
        pos=strfind(txt,'<');
        statemodel=struct('PdfClass',eval(txt(pos(2)+11:pos(3)-2))+1);
        Transition=eval(['[' txt(pos(3)+13:pos(4)-2) ']']);
        for p=2:length(pos)-3,
            Transition(p,:)=eval(['[' txt(pos(p+2)+13:pos(p+3)-2) ']']);
        end
        Transition(:,1)=Transition(:,1)+1;
        statemodel.Transition=Transition;
        phonemodel{state}=statemodel;
        txt=fgetl(fid);
        state=state+1;
    end
    for p=forphones,
        Topology{p}=phonemodel;
    end
    fgetl(fid); %</TopologyEntry>
    txt=fgetl(fid);
end
TransitionModel.Topology=Topology;
fgetl(fid); %<Triples>
txt=fgetl(fid);
Triples=[];
while ~strcmp(txt,'</Triples> '),
    Triples=[Triples; eval(['[' txt ']'])]; %columns: phone, hmm-state, pdf-id - lines: transition-state
    txt=fgetl(fid);
end
Triples(:,2:3)=Triples(:,2:3)+1;
TransitionModel.Triples=Triples;
fgetl(fid); %<LogProbs>
txt=fgetl(fid);
LogProbs=eval(txt).';
TransitionModel.LogProbs=LogProbs;
fgetl(fid); %</LogProbs>
fgetl(fid); %</TransitionModel>
fgetl(fid); %<DIMENSION> <NUMPDFS> <DiagGMM>
Pdfs=cell(1);
state=1;
while ~feof(fid),
    txt=fgetl(fid); %<GCONSTS>
    Pdfs{state}.Gconsts=eval(txt(12:end));
    txt=fgetl(fid); %<WEIGHTS>
    Pdfs{state}.Weights=eval(txt(12:end));
    fgetl(fid); %<MEANS_INVVARS>
    txt=fgetl(fid);
    Means_invvars=[];
    while isempty(strfind(txt,']')),
        Means_invvars=[Means_invvars; eval(['[' txt ']'])];
        txt=fgetl(fid);
    end
    Means_invvars=[Means_invvars; eval(['[' txt])];
    Pdfs{state}.Means_invvars=Means_invvars.';
    fgetl(fid); %<INV_VARS>
    txt=fgetl(fid);
    Inv_vars=[];
    while isempty(strfind(txt,']')),
        Inv_vars=[Inv_vars; eval(['[' txt ']'])];
        txt=fgetl(fid);
    end
    Inv_vars=[Inv_vars; eval(['[' txt])];
    Pdfs{state}.Inv_vars=Inv_vars.';
    fgetl(fid); %</DiagGMM>
    fgetl(fid); %<DiagGMM>
    state=state+1;
end
fclose(fid);
system(['rm ' filename(1:end-3) 'txt']);
State2id=zeros(size(Triples,1),1);
cur_transition_id = 1;
for tstate=1:size(Triples,1),
    State2id(tstate)=cur_transition_id;
    phone=Triples(tstate,1);
    hmm_state=Triples(tstate,2);
    my_num_ids=size(Topology{phone}{hmm_state}.Transition,1);
    cur_transition_id=cur_transition_id+my_num_ids;
end
State2id(tstate+1)=cur_transition_id;
Id2state=zeros(cur_transition_id,1);
for tstate=1:size(Triples,1),
    Id2state(State2id(tstate):State2id(tstate+1)-1)=tstate;
end
TransitionModel.State2id=State2id;
TransitionModel.Id2state=Id2state;
models.TransitionModel=TransitionModel;
models.Pdfs=Pdfs;

return