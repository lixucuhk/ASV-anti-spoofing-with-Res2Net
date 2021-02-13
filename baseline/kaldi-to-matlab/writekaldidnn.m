function writekaldidnn(nnet,filename)

% WRITEKALDIDNN Writes a neural network in Kaldi format
%
% writekaldidnn(nnet,filename)
%
% Input:
% nnet: neural network in Matlab format (see readkaldidnn for detailed
% format specification)
% filename: output filename (.DBN or .MDL extension)
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
fprintf(fid,'%s\n','<Nnet> ');
for layer=1:length(nnet),
    AffineTransform=nnet{layer}.AffineTransform;
    Activation=nnet{layer}.Activation;
    fprintf(fid,'%s %d %d \n','<AffineTransform>',AffineTransform.Size(1),AffineTransform.Size(2));
    fprintf(fid,'%s %.7g %s %.7g  [\n ','<LearnRateCoef>',AffineTransform.LearnRateCoef,'<BiasLearnRateCoef>',AffineTransform.BiasLearnRateCoef);
    for neuron=1:AffineTransform.Size(1)-1,
        fprintf(fid,' %.7g', AffineTransform.Transform(neuron,:));
        fprintf(fid,' \n ');
    end
    fprintf(fid,' %.7g', AffineTransform.Transform(end,:));
    fprintf(fid,' ]\n [');
    fprintf(fid,' %.7g', AffineTransform.Bias);
    fprintf(fid,' ]\n');
    fprintf(fid,'<%s> %d %d \n',Activation.Type,Activation.Size(1),Activation.Size(2));
end
fprintf(fid,'%s','</Nnet> ');
fclose(fid);
[~,~]=system(['rm ' filename(1:end-3) 'txt']);
[~,~]=system(['nnet-copy --binary=true ' filename(1:end-3) 'txt ' filename]);

return