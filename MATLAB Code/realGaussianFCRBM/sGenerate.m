% Version 0.100 (Unsupported, unreleased)
%
% Code provided by Graham Taylor and Geoff Hinton
%
% For more information, see:
%    http://www.cs.toronto.edu/~gwtaylor/publications/icml2009
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, expressed or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.
%
% Train a factored, conditional RBM with three-way interactions
% On labeled mocap data (10 styles, CMU subject 137)
%
% demonstrates a trained model
% generating different styles based on initialization
% dataset we use is the "clipped" CMU 137 style walks
% Assumes that demo_train.m has already been run and a model has been
% saved to the "snapshots" directory

load snapshots/audioTest_ep200.mat

%variable names needed for generation
[numdims, numfac] = size(visfac);
[numhid, junk] = size(hidfac);

n1 = nt; %preprocess2 looks at n1

preprocessAudio

numdims = size(batchdata,2); %data (visible) dimension

labeldata = [];
for jj=1:length(Labels)
  labeldata = [labeldata; Labels{jj}];
end

initdata = batchdata;

%how many frames to generate (per sequence)
numframes = 100;

%set up figure for display
% close all; h = figure(2); 
% p = get(h,'Position'); p(3) = 2*p(3); %double width
% set(h,'Position',p);

% 10 and 200
fr = 200;

% genfaccrbm
genAUDIO

 PLAYBACK = mikeSTFT(visible', FFTwindow, hop, tapWindow);
 realP = real(PLAYBACK);
% sound(real(PLAYBACK),Fs)
% a=mikeSTFT(real(PLAYBACK), FFTwindow, hop, tapWindow);
fr = 10;

% genfaccrbm
genAUDIO
 PLAYBACK2 = mikeSTFT(visible', FFTwindow, hop, tapWindow);
 realP2 = real(PLAYBACK2);
 
