% Binary audio CRBM
% Mike Junokas 1.14.16
%
% An attempt at generating a single layer gaussian to binary timbre RBM

%% PREP MODEL PARAMETERS AND DATA

clear all; close all;

%%

% imagesc(abs(SINE).^.35), axis xy % why .35?
% B = zeros(12,512)
f = 210;
Fs = 44100;
t=0:Fs*.5;   %Time vector 
w = 2; %pulse width
d1= w/2:w*210:Fs*2; %delay vector
rectPulseTrain=pulstran(t,d1,'rectpuls',w);

w2 = 2; %pulse width
d2= w/2:w*420:Fs*2; %delay vector
rectPulseTrain2=pulstran(t,d2,'rectpuls',w2);

w2 = 2; %pulse width
d2= w/2:w*840:Fs*2; %delay vector
rectPulseTrain3=pulstran(t,d2,'rectpuls',w2);


t = 0:1/Fs:1; %secondssound

 sineWave = sin(2.*pi*f.*t);
 sawWave = sawtooth(2*pi*f*t)';
 squareWave = square(2*pi*f*t)';
 triangleWave = sawtooth(2*pi*f*t, 0.5)';


FFTwindow = 210;
hop = FFTwindow;
tapWindow = ones(FFTwindow,1);
%tapWindow = hanning(FFTwindow);


Fc = 420; % Carrier frequency
s1 = sin(2*pi*t*300); % Channel 1
s2 = sawtooth(2*pi*200*t); % Channel 2
s3 = sin(2*pi*t*500);
% % % x = [s1,s2]; % Two-channel signal
dev = 1000; % Frequency deviation in modulated signal

x = fmmod(s1,Fc,Fs,175); % Modulate both channels.
% % % z = fmdemod(y,Fc,Fs,dev); % Demodulate both channels.
% % % plot(z);

x2 = fmmod(s2,500,Fs,145);

x3 = fmmod(s3,200,Fs,113);

%tapWindow = hanning(FFTwindow);
% 
chirpUp = chirp(t, 420, 1, 840)';
% chirpUp = repmat(chirpUp,16,1);
% chirpUp = chirpUp(1:44100,1);


% chirpDown =chirp(t, 210, 1, 420)';
% chirpDown = wrev(chirpDown);

% violinPizz = audioread('violinPizzG4.aif');
% violinArco = audioread('A2nonvibviolin.aif');
% kalimba = audioread('kalimbaG4.aiff');
% flute = audioread('flute.aiff');
% 
% violinArco = violinArco(:,1);
% 
% VIOLIN = mikeSTFT(violinArco, FFTwindow, hop, tapWindow)';
% KALIMBA = mikeSTFT(kalimba, FFTwindow, hop, tapWindow)';
% FLUTE = mikeSTFT(flute, FFTwindow, hop, tapWindow)';

%CHIRPUP = mikeSTFT(chirpUp, FFTwindow, hop, tapWindow)';
%CHIRPDOWN = mikeSTFT(chirpDown, FFTwindow, hop, tapWindow)';

% 
PLSTRAIN = mikeSTFT(rectPulseTrain, FFTwindow, hop, tapWindow)';
% PLSTRAIN2 = mikeSTFT(rectPulseTrain2, FFTwindow, hop, tapWindow)';
% PLSTRAIN3 = mikeSTFT(rectPulseTrain3, FFTwindow, hop, tapWindow)';



SINE = mikeSTFT(sineWave, FFTwindow, hop, tapWindow)';
% SINE2 = mikeSTFT(SINEU, FFTwindow, hop, tapWindow)';
% 
SAW = mikeSTFT(sawWave, FFTwindow,hop,tapWindow)';
% SQUARE = mikeSTFT(squareWave, FFTwindow,hop,tapWindow)';
TRI = mikeSTFT(triangleWave, FFTwindow,hop,tapWindow)';

% FREQMOD = mikeSTFT(x, FFTwindow,hop,tapWindow)';
% FREQMOD2 = mikeSTFT(x2, FFTwindow,hop,tapWindow)';
% FREQMOD3 = mikeSTFT(x3, FFTwindow,hop,tapWindow)';

% zeroMeanFeatures = bsxfun(@minus, FREQMOD, mean(FREQMOD,2));
% zeroMeanFeaturesT = zeroMeanFeatures';
% covarianceMatrix = zeroMeanFeaturesT*zeroMeanFeaturesT'/size(zeroMeanFeaturesT,2);
% 
% numberOfPrincipalComponents = 40;
% [U,v]=eigs(covarianceMatrix,numberOfPrincipalComponents);
% U = inv(sqrtm(v))*U';
% 
% IFFTPCA = mikeSTFT(U, 78, 78, hanning(78));

%%

 CRBM.classes{1,1}  = SINE(1:12,:);
  CRBM.classes{1,2}  = SAW(1:12,:);
  CRBM.classes{1,3} = TRI(1:12,:);

CRBM.data = cell2mat(CRBM.classes');

MEAN = mean(CRBM.data,1);
STD = std(CRBM.data);

%  CRBM.normData = (CRBM.data - repmat(MEAN,size(CRBM.data,1),1))./repmat(STD,size(CRBM.data,1),1);
%  CRBM.data = CRBM.normData;

CRBM.numdims = size(CRBM.data,2); %data (visible) dimension
CRBM.model = [];
CRBM.gsd = 1;

CRBM.order = 6;
CRBM.numhid = 10;
CRBM.numepochs = 2000;
CRBM.cdSteps = 1;


%% TRAINING

fprintf(1,'Training Layer 1 CRBM , order %d: %d-%d \n',CRBM.order ,CRBM.numdims,CRBM.numhid);
CRBM = train_binarycrbmMikeIdxFix2sound(CRBM);


%% GENERATING/SAMPLING DATA
% % 
  
classTested =2;
    
 numframes = 500; %how many frames to generate
 fr = 1;         %pick a starting frame from initdata                 
% % fprintf(1,'Generating %d-frame sequence of data from %d-layer CRBM ... \n',numframes);

GenLog = gen_crbmGaussMikeMin (CRBM, CRBM.classes{1,classTested}(:,:), numframes, fr);
% 
% PLAYBACK = mikeSTFT(GenLog.visible', FFTwindow, hop, tapWindow);
% sound(real(PLAYBACK),Fs)
% a=mikeSTFT(real(PLAYBACK), FFTwindow, hop, tapWindow);

 
 % rescale
% % % deNormData = repmat(STD,size(GenLog.visible,1),1).* GenLog.visible + repmat(MEAN, size(GenLog.visible,1),1);
% % %  
% % % PLAYBACK2 = mikeSTFT(deNormData', FFTwindow, hop, tapWindow);
% % % sound(real(PLAYBACK2),Fs)
% % % a2=mikeSTFT(real(PLAYBACK2), FFTwindow, hop, tapWindow);

 
