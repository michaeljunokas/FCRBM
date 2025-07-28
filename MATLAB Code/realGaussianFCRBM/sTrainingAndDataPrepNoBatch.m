% CRBM, FBM, FCRBM for Timbral Transitions - Training Script
%
% Mike Junokas 4.28.16
% 
% Code based on Graham Taylor and Geoff Hinton work at
%
%    http://www.cs.toronto.edu/~gwtaylor/publications/icml2009
%
% General data-prep, parameter setting, and training functions for
% respective boltzman machines. Use of algorithm can be set in the bottom
% sections.
%
% Unlike the original Taylor code, this script does not generate
% mini-batches of the data; can be optimized in the future but currently is
% the only way to get the algorithms to work with spectral material
%
%
% For working examples and documentation, see 'FCRBMdoc.txt'

clear all; close all;
more off;   %turn off paging

f = 400;
Fs = 44100;
t = 0:1/Fs:5; %secondssound

FFTwindow = 4000;
hop = FFTwindow;
tapWindow = ones(FFTwindow,1);
%tapWindow = hamming(FFTwindow);

Fc = 420; % Carrier frequency
s1 = sin(2*pi*t*200); % Channel 1
s2 = sawtooth(2*pi*200*t); % Channel 2
s3 = sin(2*pi*t*400);
s4 = sawtooth(2*pi*150*t,.5);
dev = 1000; % Frequency deviation in modulated signal

x = fmmod(s1,Fc,Fs,175); % Modulate both channels.
x2 = fmmod(s2,500,Fs,145);
x3 = fmmod(s3,200,Fs,113);
x4 = fmmod(s4,100,Fs,150);

c1 = x(1,1:2000);
c2 = x(1,1:2000);
C1 = [c1,c2];
combo1 = repmat(C1, 1, 40);

c3 = x3(1,1:4000);
c4 = x4(1,1:4000);
C2 = [c3,c4];
combo2 = repmat(C2, 1, 20);

Audio{1,1} = mikeSTFT(combo1, FFTwindow, hop, tapWindow)';

Audio{1,2}  = mikeSTFT(combo2, FFTwindow, hop, tapWindow)';

combinedData = vertcat(Audio{1},Audio{2});


% Normalize...Standard

% data_mean = mean(combinedData,1);
% data_std = std(combinedData);
% normData =( combinedData - repmat(data_mean,size(combinedData,1),1) ) ./ ...
%   repmat( data_std, size(combinedData,1),1);

% By FFT bin 1....
% fftMean = combinedData(:,:) ./ repmat(combinedData(:,1),1, size(combinedData,2));

% combinedData = fftMean;

% 
% Audio{1,1} = repmat(.9, 100, 100);
% 
% Audio{1,2}  = repmat(.1, 100, 100);
% 
% combinedData = vertcat(Audio{1},Audio{2});


Labels{1} = repmat([1,0], size(Audio{1,1},1), 1);
Labels{2}=repmat([0,1], size(Audio{1,2},1), 1);
allLabels = vertcat(Labels{:,1},Labels{:,2});


% order
nt = 10; 

numdims = size(Audio{1},2);

for ii = 1:length(Audio)
    classLength(ii) = size(Audio{ii},1);
    if ii == 1
        classIndex = classLength(ii);
    else
        classIndex = [classIndex classIndex(end)+classLength(ii)];
    end
end

for i = 1:length(classIndex)
    range{i} = classIndex(i) - classLength(i) + nt+1:classIndex(i);
end

INDICES = [range{:}];
numcases = length(INDICES);   


past = zeros(numcases,nt*numdims);
data = combinedData(INDICES,:);
labels = allLabels(INDICES,:);

for hh=nt:-1:1 %note reverse order
      past(:,numdims*(nt-hh)+1:numdims*(nt-hh+1)) = combinedData(INDICES-hh,:) + randn(numcases,numdims);
end      
    

%Training properties
numhid = 4;

numlabels = size(Labels,2); % labels? instead of Labels?

%There are three types of factors
%But for now, we will set them all to be size numfac
numfac = 4; 
numfeat = 2; %number of distributed "style" features
maxepoch = 2500;
cdsteps = 1;
pastnoise = 1;

snapshotevery=100; %write out a snapshot of the weights every xx epochs


fprintf(1,'Training Layer 1 FBM, order %d: %d-%d(%d) \n',nt,numdims, ...
  numhid,numfac);
restart=1;      %initialize weights
%%
%train network with only feature-to-factor weights tied

 gaussianfaccrbmAUDIOnoBatch % WORKS
% gaussFBM% gaussianfbmAUDIOnoBatch %DOES NOT WORK;

% gaussianfbm_sharefeatfacAUDIOnoBatch % DOES NOT WORK; LABELS?

fprintf(1,'Training finished. Run demo_generate to generate data\n');
