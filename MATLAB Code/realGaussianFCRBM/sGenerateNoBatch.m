% CRBM, FBM, FCRBM for Timbral Transitions - Generation Script
%
% Mike Junokas 4.28.16
% 
% Code based on Graham Taylor and Geoff Hinton work at
%
%    http://www.cs.toronto.edu/~gwtaylor/publications/icml2009
%
% General labeling and generation functions for respective boltzman 
% machines. Distinct samples must be init from different indexes found in
% the 'data'
%
% Unlike the original Taylor code, this script does not generate
% mini-batches of the data; can be optimized in the future but currently is
% the only way to get the algorithms to work with spectral material
%
%
% For working examples and documentation, see 'FCRBMdoc.txt'
%
%AudioEXTRA  = mikeSTFT(s2, FFTwindow, hop, tapWindow)';
n1=nt;
labeldata = [];
for jj=1:length(Labels)
  labeldata = [labeldata; Labels{jj}];
end

initdata = data;

%how many frames to generate (per sequence)
numframes = 50;
numhid1=numhid;


%%%%%% FUNCTION FOR FIRST SOUND IDX
fr = 2;
genfaccrbm

%genFBM %genfaccrbm %NO LABELS

%newdata = repmat(data_std, size(visible,1),1) .*  visible + repmat(data_mean, size(visible,1),1);

combo1 = visible;
 PLAYBACK = mikeSTFT(visible', FFTwindow, hop, tapWindow);
 realP = real(PLAYBACK);
% % %%%%%%

fr = 40;
genfaccrbm

combo2 = visible;
 PLAYBACK = mikeSTFT(visible', FFTwindow, hop, tapWindow);
 realP = real(PLAYBACK);
% % %%%%%%



COMBOD = [combo1;combo2];

PLAYBACKCOMBOD = mikeSTFT(COMBOD', FFTwindow, hop, tapWindow);
realZ = real(PLAYBACKCOMBOD);

sound(realZ,Fs)

COMBODcopy = COMBOD;

PLAYBACKCOMBOD = mikeSTFT(COMBODcopy', FFTwindow, hop, tapWindow);
realD = real(PLAYBACKCOMBOD);

point1 = COMBOD(1,:);
point2 = COMBOD(52,:);

A = arrayfun(@(x,y) linspace(x,y,10), point1, point2, 'uni',0);
B= [A{:}]
C = reshape(B, 10, 2001)

D= [combo1;C;combo2];

PLAYBACKCOMBOD = mikeSTFT(D', FFTwindow, hop, tapWindow);
realD = real(PLAYBACKCOMBOD);

