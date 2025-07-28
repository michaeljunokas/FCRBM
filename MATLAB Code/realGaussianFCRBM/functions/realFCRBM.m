% Binary audio fCRBM
% Mike Junokas 1.28.16
%
% An attempt at generating a single layer binary audio FCRBM

%% PREP MODEL PARAMETERS AND DATA

clear all; close all;

f = 210;
Fs = 44100;
t = 0:1/Fs:5; %secondssound

dt = 1/Fs;


sineWave = sin(2.*pi*f.*t)';
sawWave = sawtooth(2*pi*f*t)';
squareWave = square(2*pi*f*t)';
triangleWave = sawtooth(2*pi*f*t, 0.5)';

FFTwindow = 2000;
hop = FFTwindow;
tapWindow = ones(FFTwindow,1);
%tapWindow = hamming(FFTwindow);

Fc = 420; % Carrier frequency
s1 = sin(2*pi*t*300); % Channel 1
s2 = sawtooth(2*pi*200*t); % Channel 2
s3 = sin(2*pi*t*500);
s4 = sawtooth(2*pi*150*t,.5);
% % % x = [s1,s2]; % Two-channel signal
dev = 1000; % Frequency deviation in modulated signal

x = fmmod(s1,Fc,Fs,175); % Modulate both channels.
% % % z = fmdemod(y,Fc,Fs,dev); % Demodulate both channels.
% % % plot(z);

x2 = fmmod(s2,500,Fs,145);

x3 = fmmod(s3,200,Fs,113);

x4 = fmmod(s4,100,Fs,150);

c1 = x(1,1:2000);
c2 = x(1,1:2000);
C1 = [c1,c2];
combo1 = repmat(C1, 1, 20);

c3 = x3(1,1:4000);
c4 = x4(1,1:4000);
C2 = [c3,c4];
combo2 = repmat(C2, 1, 10);



 SINE = mikeSTFT(sineWave, FFTwindow, hop, tapWindow)';
 SAW = mikeSTFT(sawWave, FFTwindow,hop,tapWindow)';
 TRI = mikeSTFT(triangleWave, FFTwindow,hop,tapWindow)';

   FCRBM.classes{1,1} = mikeSTFT(combo1, FFTwindow, hop, tapWindow)';
  FCRBM.classes{1,2}  = mikeSTFT(combo2, FFTwindow, hop, tapWindow)';
% 

% FCRBM.classes{1,1} = TRI(:,2:end)';
% FCRBM.classes{1,2} = SAW(:,2,end)';
% FCRBM.classes{1,3} = SINE(:,2,end);
% 
% 
% label1 = repmat([1,0], size(FCRBM.classes{1,1},1), 1);
% label2 =repmat([0,1], size(FCRBM.classes{1,2},1), 1);
% label3 =repmat([0,0,1], size(FCRBM.classes{1,3},1), 1);
% FCRBM.labels = vertcat(label1, label2);

% 
% FCRBM.data = cell2mat(FCRBM.classes');
% MEAN = mean(FCRBM.data,1);
% STD = std(FCRBM.data);
% FCRBM.normData = (FCRBM.data - repmat(MEAN,size(FCRBM.data,1),1))./repmat(STD,size(FCRBM.data,1),1);
% FCRBM.data = FCRBM.normData;
% 
% meanS = mean(sineWave);
% 
% normSin = mikeSTFT((sineWave-meanS), FFTwindow, hop, tapWindow)/840;

% SINE = mikeSTFT(sineWave, FFTwindow, hop, tapWindow);
% SAW = mikeSTFT(sawWave,FFTwindow,hop,tapWindow);
% 
% FCRBM.classes{1,1} = SAW;  %  Watch values so that it doesn't overlap fft
% FCRBM.classes{1,2} = SINE;

% B = mikeSTFT(SINE,FFTwindow,hop,tapWindow)*Fs;


% COMBOFFT = mikeSTFT(vertcat(sineWave,triangleWave), FFTwindow,hop,tapWindow)';
% % dc = COMBOFFT(:,1);
% % normCombo = COMBOFFT./repmat(dc,1,size(COMBOFFT,2));
% normCombo = normCombo(:,2:end);
% 
% maxNormReal = max(real(COMBOFFT(:)));
% maxNormImag = max(imag(COMBOFFT(:)));
% maxNorm = max(COMBOFFT(:));
% % 
% NORM = normCombo/maxNorm;
% Normed = normCombo/maxNormReal;
% Normedi = normCombo/maxNormImag;
% 
% FCRBM.classes{1,1} = Normed(1:262-1,:);  %  Watch values so that it doesn't overlap fft
% FCRBM.classes{1,2} = Normed(264:525-1,:);
% FCRBM.classes{1,3} = COMBOFFT(527:end,2:end);

FCRBM.data = cell2mat(FCRBM.classes'); % Transposing to keep observations as rows; cell2mat problem?
% % MEAN = mean(FCRBM.data(:));
% % STD = std(FCRBM.data);
% % FCRBM.normData = (FCRBM.data - repmat(MEAN,size(FCRBM.data,1),size(FCRBM.data,2)))./repmat(STD,size(FCRBM.data,1),1);
% % FCRBM.data = FCRBM.normData;


label1 = repmat([1,0], size(FCRBM.classes{1,1},1), 1);
label2 = repmat([0,1], size(FCRBM.classes{1,2},1), 1);

FCRBM.labels = vertcat(label1,label2);

% % % COMBOFFT = mikeSTFT(vertcat(sineWave,sawWave,triangleWave), FFTwindow,hop,tapWindow)';
% % % dc = COMBOFFT(:,1);
% % % normCombo = COMBOFFT./repmat(dc,1,size(COMBOFFT,2));
% % % 
% % % FCRBM.classes{1,1} = COMBOFFT(1:262-1,2:end);  %  Watch values so that it doesn't overlap fft
% % % FCRBM.classes{1,2} = COMBOFFT(264:525-1,2:end);
% % % FCRBM.classes{1,3} = COMBOFFT(527:end,2:end);
% % % 
% % % FCRBM.data = cell2mat(FCRBM.classes'); % Transposing to keep observations as rows; cell2mat problem?
% % % 
% % % label1 = repmat([1,0,0], size(FCRBM.classes{1,1},1), 1);
% % % label2 = repmat([0,1,0], size(FCRBM.classes{1,2},1), 1);
% % % label3 = repmat([0,0,1], size(FCRBM.classes{1,3},1), 1);
% % % 
% % % FCRBM.labels = vertcat(label1,label2,label3);

FCRBM.order = 2;
FCRBM.numhid = 2;
FCRBM.numfac = 20;
FCRBM.numfeat = 10;
FCRBM.numepochs = 20;
FCRBM.numdims = size(FCRBM.data,2); %data (visible) dimension
FCRBM.cdSteps = 1;
FCRBM.model = [];

%% TRAINING
 
fprintf(1,'Training Layer 1 CRBM , order %d: %d-%d \n',FCRBM.order ,FCRBM.numdims,FCRBM.numhid);
FCRBM = MikeTestFCRBMrealTEST(FCRBM);

%% GENERATING/SAMPLING DATA
% % 
  
numClasses = length(FCRBM.classes);
classTested =1;
numframes = 20;
fr = 1;                       
labels =zeros(numframes,numClasses);
labels(:,classTested) = 1;


%gen_sharefeatfac

GenLog = MikeTestFCRBMgen (FCRBM, FCRBM.classes{1,classTested}(:,:), labels, numframes, fr);
 
% addDC = GenLog.visible'.*repmat(dc,1,size(GenLog.visible',2));
% 
% 
% PLAYBACK = mikeSTFT(GenLog.visible', FFTwindow, hop, tapWindow);
% sound(real(PLAYBACK),Fs)
% a=mikeSTFT(real(PLAYBACK), FFTwindow, hop, tapWindow);
% 
%  % rescale
% deNormData = repmat(STD,size(GenLog.visible,1),1).* GenLog.visible + repmat(MEAN, size(GenLog.visible,1),1);
%  
% PLAYBACK2 = mikeSTFT(deNormData', FFTwindow, hop, tapWindow);
% sound(real(PLAYBACK2),Fs)
% a2=mikeSTFT(real(PLAYBACK2), FFTwindow, hop, tapWindow);
