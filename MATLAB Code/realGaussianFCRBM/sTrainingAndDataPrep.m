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

clear all; close all;
more off;   %turn off paging

f = 210;
Fs = 44100;
t = 0:1/Fs:5; %secondssound

FFTwindow = 2000;
hop = FFTwindow;
tapWindow = ones(FFTwindow,1);
%tapWindow = hamming(FFTwindow);

Fc = 420; % Carrier frequency
s1 = sin(2*pi*t*300); % Channel 1
s2 = sawtooth(2*pi*200*t); % Channel 2
s3 = sin(2*pi*t*500);
s4 = sawtooth(2*pi*150*t,.5);
dev = 1000; % Frequency deviation in modulated signal

x = fmmod(s1,Fc,Fs,175); % Modulate both channels.
x2 = fmmod(s2,500,Fs,145);
x3 = fmmod(s3,200,Fs,113);
x4 = fmmod(s4,100,Fs,150);

c1 = x(1,1:2000);
c2 = x(1,1:2000);
C1 = [c1,c2];
combo1 = repmat(C1, 1, 100);

c3 = x3(1,1:4000);
c4 = x4(1,1:4000);
C2 = [c3,c4];
combo2 = repmat(C2, 1, 50);

Audio{1,1} = mikeSTFT(combo1, FFTwindow, hop, tapWindow)';
Audio{1,2}  = mikeSTFT(combo2, FFTwindow, hop, tapWindow)';

Labels{1} = repmat([1,0], size(Audio{1,1},1), 1);
Labels{2}=repmat([0,1], size(Audio{1,2},1), 1);

% order
n1 = 2; 
 

%%%% FOR BATCH

%Run the 2nd stage of pre-processing
%This drops the zero/constant dimensions and builds mini-batches
preprocessAudio

numdims = size(batchdata,2); %data (visible) dimension

labeldata = [];
for jj=1:length(Labels)
  labeldata = [labeldata; Labels{jj}];
end

initdata = batchdata;

%%% FOR NO BATCH




%Training properties
numhid1 = 2;
%There are three types of factors
%But for now, we will set them all to be size numfac
numfac = 1; 
numfeat = 1; %number of distributed "style" features
maxepoch = 250;
cdsteps = 1;
pastnoise = 1;

%every xxx epochs, write a snapshot of the model
%will be written to snapshot_path_epxxx.mat
snapshot_path = 'snapshots/audioTest'
%snapshot_path = 'default' %don't overwrite our models
% 
snapshotevery=100; %write out a snapshot of the weights every xx epochs

nt=n1;
numhid=numhid1;

fprintf(1,'Training Layer 1 FBM, order %d: %d-%d(%d) \n',nt,numdims, ...
  numhid,numfac);
restart=1;      %initialize weights
%%
%train network with only feature-to-factor weights tied

gaussianfaccrbmAUDIO
% gaussianfbm_sharefeatfacAUDIO

fprintf(1,'Training finished. Run demo_generate to generate data\n');
