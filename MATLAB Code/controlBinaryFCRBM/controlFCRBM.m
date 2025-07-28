% Binary audio fCRBM
% Mike Junokas 1.28.16
%
% An attempt at generating a single layer binary audio FCRBM

%% PREP MODEL PARAMETERS AND DATA

clear all; close all;


FCRBM.classes{1,1} = [[1],[1],[1]]';
FCRBM.classes{1,2} = [[0],[0],[0]]';

FCRBM.labels = vertcat([1,0],[1,0],[1,0],[0,1],[0,1],[0,1]);


FCRBM.data = cell2mat(FCRBM.classes');
FCRBM.order = 2;
FCRBM.numhid = 2;
FCRBM.numfac = 2;
FCRBM.numfeat = 2;
FCRBM.numepochs = 500;
FCRBM.numdims = size(FCRBM.data,2); %data (visible) dimension
FCRBM.model = [];

%% TRAINING

fprintf(1,'Training Layer 1 CRBM , order %d: %d-%d \n',FCRBM.order ,FCRBM.numdims,FCRBM.numhid);
FCRBM = MikeTestFCRBM(FCRBM);

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
 
