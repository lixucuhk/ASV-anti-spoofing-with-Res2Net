function nnet=readkaldidnn(filename)

% READKALDIDNN Reads a neural network in Kaldi format
%
% nnet=readkaldidnn(filename)
%
% Input:
% filename: Kaldi neural network filename (.DBN or .MDL extension)
%
% Output:
% nnet: Matlab cell with the following fields for each layer l:
% * nnet{l}.AffineTransform.Size: size of the weight matrix
% * nnet{l}.AffineTransform.LearnRateCoef: learning rate for the weights
% * nnet{l}.AffineTransform.BiasLearnRateCoef: learning rate for the bias
% * nnet{l}.AffineTransform.Transform: weight matrix
% * nnet{l}.AffineTransform.Bias: bias vector
% * nnet{l}.Activation.Type: nonlinear transform type (string)
% * nnet{l}.Activation.Size: size of the nonlinear transform
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

[~,~]=system(['nnet-copy --binary=false ' filename ' ' filename(1:end-3) 'txt']);
fid=fopen([filename(1:end-3) 'txt'],'r');
fgetl(fid); %<Nnet>
txt=fgetl(fid); %<AffineTransform> XXX YYY
nnet=cell(1);
layer=0;
while ~strcmp(txt,'</Nnet> '),
    layer=layer+1;
    AffineTransform.Size=eval(['[' txt(19:end) ']']);
    txt=fgetl(fid); %<LearnRateCoef> 1 <BiasLearnRateCoef> 1  [
    pos=strfind(txt,'<');
    AffineTransform.LearnRateCoef=eval(txt(17:pos(2)-2));
    AffineTransform.BiasLearnRateCoef=eval(txt(pos(2)+20:end-3));
    AffineTransform.Transform=zeros(AffineTransform.Size(1),AffineTransform.Size(2));
    for neuron=1:AffineTransform.Size(1)-1,
        txt=fgetl(fid);
        AffineTransform.Transform(neuron,:)=eval(['[' txt ']']);
    end
    txt=fgetl(fid);
    AffineTransform.Transform(AffineTransform.Size(1),:)=eval(['[' txt]);
    txt=fgetl(fid);
    AffineTransform.Bias=eval(txt);
    txt=fgetl(fid);
    pos=strfind(txt,'>');
    Activation.Type=txt(2:pos-1);
    Activation.Size=eval(['[' txt(pos+2:end) ']']);
    nnet{layer}.AffineTransform=AffineTransform;
    nnet{layer}.Activation=Activation;
    txt=fgetl(fid);
end
fclose(fid);
[~,~]=system(['rm ' filename(1:end-3) 'txt']);

return