% Binary audio CRBM
% Mike Junokas 10.20.15
%
% An attempt at generating a single layer binary audio RBM

%% PREP MODEL PARAMETERS AND DATA

clear all; close all;
%%
% parameters(1,:) = [4 100 4 500 10]; % order, batchsize, hidunits, epochs, CD steps

%   CRBM.classes{1,1}  = [0 1 0 1 0 1 0 1]'; 
%   CRBM.classes{1,2}  = [1 0 1 0 1 0 1 0]';
%  
% CRBM.classes{1,3}  = [0 0 0 0 0 0 0]';
% CRBM.classes{1,4}  = [1 1 1 1 1 1 1]';


%  CRBM.classes{1,1}  = [[0 0 0 0 0 0 0];[1 1 1 1 1 1 1]]';
%  CRBM.classes{1,2}  = [[1 1 1 1 1 1 1];[0 0 0 0 0 0 0]]';
% 
%  CRBM.classes{1,1}  = [0 1 0]';
%  CRBM.classes{1,2}  = [1 0 1]';
%  
% a = diag([1 1 1 1 1 1 1 1 1 1 1 1]);
% a = vertcat(a, fliplr(a));
% 
% CRBM.classes{1,1} = a;
% CRBM.classes{1,2} = zeros(24,12);

% CRBM.classes{1,3} = [[1;1],[1;1],[1;1]]';


                  
 class1 = [ [1 0 0 0 0 0 0]',[0 0 0 1 0 0 0]',[1 0 0 0 0 0 0]']';
  class2 = [ [1 0 0 0 0 0 0]',[0 0 0 0 1 0 0]',[1 0 0 0 0 0 0]']';                 
 class3 = [ [1 0 0 0 0 0 0]',[0 0 0 0 0 0 1]',[1 0 0 0 0 0 0]']';
 class4 = [[1 0 0 0 0 0 0]',[0 1 0 0 0 0 0]', [0 0 0 0 1 0 0]',[1 0 0 0 0 0 0]']';
 class5 = [[1 0 0 0 0 0 0]',[0 1 0 0 0 0 0]', [0 0 0 0 0 0 1]',[1 0 0 0 0 0 0]']';
class6 = [[1 0 0 0 0 0 0]',[0 0 1 0 0 0 0]', [0 0 0 1 0 0 0]',[1 0 0 0 0 0 0]']';     
 class7 = [[1 0 0 0 0 0 0]',[0 0 0 1 0 0 0]', [0 0 0 0 1 0 0]',[1 0 0 0 0 0 0]']';
 class8 = [[1 0 0 0 0 0 0]',[0 0 0 1 0 0 0]', [0 0 0 0 0 0 1]',[1 0 0 0 0 0 0]']';
class9 = [[1 0 0 0 0 0 0]',[0 0 0 0 1 0 0]', [0 0 0 0 0 0 1]',[1 0 0 0 0 0 0]']';
class10 = [[1 0 0 0 0 0 0]',[0 0 0 0 0 1 0]', [0 0 0 1 0 0 0]',[1 0 0 0 0 0 0]']';
class11 = [[1 0 0 0 0 0 0]',[0 0 0 0 0 0 1]', [0 0 0 0 1 0 0]',[1 0 0 0 0 0 0]']';

% class12 = [[0 1 0 0 0 0 0]',[0 0 0 1 0 0 0]',[1 0 0 0 0 0 0]']';
% class13 = [[0 1 0 0 0 0 0]',[0 0 0 0 0 0 1]',[1 0 0 0 0 0 0]']';
% class14 = [[0 0 1 0 0 0 0]',[0 0 0 1 0 0 0]',[1 0 0 0 0 0 0]']';
% class15 = [[0 0 0 1 0 0 0]',[0 0 0 0 1 0 0]',[1 0 0 0 0 0 0]']';
% class16 = [[0 0 0 1 0 0 0]',[0 0 0 0 0 0 1]',[1 0 0 0 0 0 0]']';
% class17 = [[0 0 0 0 1 0 0]',[0 0 0 0 0 0 1]',[1 0 0 0 0 0 0]']';
% class18 = [[0 0 0 0 0 1 0]',[0 0 0 1 0 0 0]',[1 0 0 0 0 0 0]']';
% class19 = [[0 0 0 0 0 0 1]',[0 0 0 0 1 0 0]',[1 0 0 0 0 0 0]']';

% class20 = [ [0 1 0 0 0 0 0]',[0 0 0 1 0 0 0]',[1 0 0 0 0 0 0]']';
% class21 = [ [0 1 0 0 0 0 0]',[0 0 0 0 0 0 1]',[1 0 0 0 0 0 0]']';
% class22 = [ [0 0 1 0 0 0 0]',[0 0 0 1 0 0 0]',[1 0 0 0 0 0 0]']';
% class23 = [ [0 0 0 1 0 0 0]',[0 0 0 0 1 0 0]',[1 0 0 0 0 0 0]']';
% class24 = [ [0 0 0 1 0 0 0]',[0 0 0 0 0 0 1]',[1 0 0 0 0 0 0]']';
% class25 = [ [0 0 0 0 1 0 0]',[0 0 0 0 0 0 1]',[1 0 0 0 0 0 0]']';
% class26 = [ [0 0 0 0 0 1 0]',[0 0 0 1 0 0 0]',[1 0 0 0 0 0 0]']';
% class27 = [ [0 0 0 0 0 0 1]',[0 0 0 0 1 0 0]',[1 0 0 0 0 0 0]']';

% class12 = [ [1 0 0 0 0 0 0]',[0 0 0 0 0 1 0]',[0 1 0 0 0 0 0]', [0 0 0 0 1 0 0]',[1 0 0 0 0 0 0]']';
% class13 = [[1 0 0 0 0 0 0]', [0 0 0 0 0 1 0]',[0 1 0 0 0 0 0]', [0 0 0 0 0 0 1]',[1 0 0 0 0 0 0]']';
% class14 = [[1 0 0 0 0 0 0]',[0 0 0 0 0 1 0]',[0 0 0 1 0 0 0]', [0 0 0 0 1 0 0]',[1 0 0 0 0 0 0]']';
% class15 = [ [1 0 0 0 0 0 0]',[0 0 1 0 0 0 0]',[0 0 0 1 0 0 0]', [0 0 0 0 1 0 0]',[1 0 0 0 0 0 0]']';
% class16 = [[1 0 0 0 0 0 0]',[0 0 0 0 0 1 0]',[0 0 0 1 0 0 0]', [0 0 0 0 0 0 1]',[1 0 0 0 0 0 0]']';
% class17 = [[1 0 0 0 0 0 0]', [0 0 1 0 0 0 0]',[0 0 0 1 0 0 0]', [0 0 0 0 0 0 1]',[1 0 0 0 0 0 0]']';
% class18 = [ [1 0 0 0 0 0 0]',[0 0 0 1 0 0 0]',[0 0 0 0 1 0 0]', [0 0 0 0 0 0 1]',[1 0 0 0 0 0 0]']';
% class19 = [[1 0 0 0 0 0 0]', [0 1 0 0 0 0 0]',[0 0 0 0 1 0 0]', [0 0 0 0 0 0 1]',[1 0 0 0 0 0 0]']';
% class20 = [[1 0 0 0 0 0 0]',[0 0 1 0 0 0 0]',[0 0 0 0 0 1 0]', [0 0 0 1 0 0 0]',[1 0 0 0 0 0 0]']';
% class21 = [ [1 0 0 0 0 0 0]',[0 0 0 0 0 0 1]',[0 0 0 0 0 1 0]', [0 0 0 1 0 0 0]',[1 0 0 0 0 0 0]']';
% class22 = [[1 0 0 0 0 0 0]', [0 0 0 0 1 0 0]',[0 0 0 0 0 1 0]', [0 0 0 1 0 0 0]',[1 0 0 0 0 0 0]']';
% class23 = [[1 0 0 0 0 0 0]', [0 0 0 1 0 0 0]',[0 0 0 0 0 0 1]', [0 0 0 0 1 0 0]',[1 0 0 0 0 0 0]']';
% class24 = [[1 0 0 0 0 0 0]', [0 1 0 0 0 0 0]',[0 0 0 0 0 0 1]', [0 0 0 0 1 0 0]',[1 0 0 0 0 0 0]']';
% 
% class33 = [ [1 0 0 0 0 0 0]',[0 0 0 0 0 1 0]',[0 1 0 0 0 0 0]', [0 0 0 0 1 0 0]']';
% class34 = [ [1 0 0 0 0 0 0]',[0 0 0 0 0 1 0]',[0 1 0 0 0 0 0]', [0 0 0 0 0 0 1]']';
% class35 = [[1 0 0 0 0 0 0]',[0 0 0 0 0 1 0]',[0 0 0 1 0 0 0]', [0 0 0 0 1 0 0]']';
% class36 = [ [1 0 0 0 0 0 0]',[0 0 1 0 0 0 0]',[0 0 0 1 0 0 0]', [0 0 0 0 1 0 0]']';
% class37 = [[1 0 0 0 0 0 0]',[0 0 0 0 0 1 0]',[0 0 0 1 0 0 0]', [0 0 0 0 0 0 1]']';
% class38 = [ [1 0 0 0 0 0 0]',[0 0 1 0 0 0 0]',[0 0 0 1 0 0 0]', [0 0 0 0 0 0 1]']';
% class39 = [ [1 0 0 0 0 0 0]',[0 0 0 1 0 0 0]',[0 0 0 0 1 0 0]', [0 0 0 0 0 0 1]']';
% class40 = [ [1 0 0 0 0 0 0]',[0 1 0 0 0 0 0]',[0 0 0 0 1 0 0]', [0 0 0 0 0 0 1]']';
% class41 = [[1 0 0 0 0 0 0]',[0 0 1 0 0 0 0]',[0 0 0 0 0 1 0]', [0 0 0 1 0 0 0]']';
% class42 = [ [1 0 0 0 0 0 0]',[0 0 0 0 0 0 1]',[0 0 0 0 0 1 0]', [0 0 0 1 0 0 0]']';
% class43 = [[1 0 0 0 0 0 0]', [0 0 0 0 1 0 0]',[0 0 0 0 0 1 0]', [0 0 0 1 0 0 0]']';
% class44 = [ [1 0 0 0 0 0 0]',[0 0 0 1 0 0 0]',[0 0 0 0 0 0 1]', [0 0 0 0 1 0 0]']';
% class45 = [[1 0 0 0 0 0 0]', [0 1 0 0 0 0 0]',[0 0 0 0 0 0 1]', [0 0 0 0 1 0 0]']';


% %  class11 = [[0 0 0 0 0 0 0]', [0 0 0 0 0 0 0]',[0 0 0 0 0 0 0]', [0 0 0 0 0 0 0]']';
%  class4 = [[1 0 0 0 0 0 0]', [0 0 0 1 0 0 0]', [1 0 0 0 0 0 0]', [0 0 0 0 1 0 0]']';
%  class5 = [[1 0 0 0 0 0 0]', [0 0 0 1 0 0 0]', [1 0 0 0 0 0 0]', [0 0 0 1 0 0 0]']';
%  class6 = [[1 0 0 0 0 0 0]', [0 0 0 0 0 0 1]', [1 0 0 0 0 0 0]', [0 0 0 0 0 0 1]']';
%  
%  class7 = [[1 0 0 0 0 0 0]', [0 1 0 0 0 0 0]', [0 0 0 0 1 0 0]']';
%  class8 = [[1 0 0 0 0 0 0]', [0 1 0 0 0 0 0]', [0 0 0 0 0 0 1]']';
%  class9 = [[1 0 0 0 0 0 0]', [0 0 1 0 0 0 0]', [0 0 0 1 0 0 0]']';
%  class10 = [[1 0 0 0 0 0 0]', [0 0 0 1 0 0 0]', [0 0 0 0 1 0 0]']';
%  class11 = [[1 0 0 0 0 0 0]', [0 0 0 1 0 0 0]', [0 0 0 0 0 0 1]']';
%  class12 = [[1 0 0 0 0 0 0]', [0 0 0 0 1 0 0]', [0 0 0 0 0 0 1]']';
%  class13 = [[1 0 0 0 0 0 0]', [0 0 0 0 0 1 0]', [0 0 0 1 0 0 0]']';
%  class14 = [[1 0 0 0 0 0 0]', [0 0 0 0 0 0 1]', [0 0 0 0 1 0 0]']';

  REPEAT = 3;
%  
%  CRBM.classes{1,1} = repmat(classBach,REPEAT,1);
%   CRBM.classes{1,2} = repmat(classBach2,REPEAT,1)
%    CRBM.classes{1,3} = repmat(classBach3,REPEAT,1);
%     CRBM.classes{1,4} = repmat(classBach4,REPEAT,1);
%      CRBM.classes{1,5} = repmat(classBach5,REPEAT,1);
%       CRBM.classes{1,6} = repmat(classBach6,REPEAT,1);
  
CRBM.classes{1,1} = repmat(class1, REPEAT ,1);
CRBM.classes{1,2} = repmat(class2, REPEAT ,1);
   CRBM.classes{1,3}  = repmat(class3,REPEAT ,1);
  CRBM.classes{1,4}  = repmat(class4, REPEAT ,1);
   CRBM.classes{1,5} = repmat(class5, REPEAT ,1);
  CRBM.classes{1,6} = repmat(class6, REPEAT ,1);
   CRBM.classes{1,7} = repmat(class7, REPEAT ,1);
  CRBM.classes{1,8} = repmat(class8, REPEAT ,1);
   CRBM.classes{1,9} = repmat(class9, REPEAT ,1);
  CRBM.classes{1,9} = repmat(class10, REPEAT ,1);
      CRBM.classes{1,11} = repmat(class11, REPEAT ,1); 
%     CRBM.classes{1,12} = repmat(class12, REPEAT ,1);
%   CRBM.classes{1,13} = repmat(class13, REPEAT ,1);
%    CRBM.classes{1,14} = repmat(class14,REPEAT ,1);
%   CRBM.classes{1,15} = repmat(class15, REPEAT ,1);
%      CRBM.classes{1,16} = repmat(class16, REPEAT ,1); 
%    CRBM.classes{1,17} = repmat(class17, REPEAT ,1);
%   CRBM.classes{1,18}= repmat(class18, REPEAT ,1);
%    CRBM.classes{1,19} = repmat(class19,REPEAT ,1);
%    CRBM.classes{1,20} = repmat(class20, REPEAT ,1);
%   CRBM.classes{1,21} = repmat(class21, REPEAT ,1);
%    CRBM.classes{1,22} = repmat(class22,REPEAT ,1);
%   CRBM.classes{1,23} = repmat(class23, REPEAT ,1);
%      CRBM.classes{1,24} = repmat(class24, REPEAT ,1); 
%    CRBM.classes{1,25} = repmat(class25, 1 ,1);
%   CRBM.classes{1,26} = repmat(class26, 1 ,1);
%    CRBM.classes{1,27}  = repmat(class27,1 ,1);
%       CRBM.classes{1,28} = repmat(class28,1 ,1);
%      CRBM.classes{1,29} = repmat(class29, 1 ,1); 
%     CRBM.classes{1,30} = repmat(class30, 1 ,1);
%   CRBM.classes{1,31}= repmat(class31, 1 ,1);
%    CRBM.classes{1,32} = repmat(class32,1 ,1);
%    CRBM.classes{1,33} = repmat(class33, 2 ,1);
%   CRBM.classes{1,34} = repmat(class34, 2 ,1);
%    CRBM.classes{1,35} = repmat(class35,2 ,1);
%   CRBM.classes{1,36} = repmat(class36, 2 ,1);
%       CRBM.classes{1,37}  = repmat(class37, 2 ,1); 
%    CRBM.classes{1,38}  = repmat(class38, 2 ,1);
%   CRBM.classes{1,39}= repmat(class39, 2 ,1);
%    CRBM.classes{1,40} = repmat(class40,2 ,1);
%     CRBM.classes{1,41} = repmat(class41,2 ,1);
%   CRBM.classes{1,42} = repmat(class42, 2 ,1);
%       CRBM.classes{1,43}  = repmat(class43, 2 ,1); 
%    CRBM.classes{1,44}  = repmat(class44, 2 ,1);
%   CRBM.classes{1,45}= repmat(class45, 2 ,1);

 
 
 
%  CRBM.classes{1,18} = mul18;
%  CRBM.classes{1,19} = mul19;
 
%  CRBM.classes{1,1} = [   [1 0 0 1 0 0 1 0 0]',...
%                          [1 0 0 1 0 0 1 0 0]',...
%                          [1 0 0 1 0 0 1 0 0]']';
%                      
%  CRBM.classes{1,2} = [   [1 0 0 1 0 0 0 1 0]',...
%                          [1 0 0 1 0 0 0 1 0]',...
%                          [1 0 0 1 0 0 0 1 0]']';
%                       
%  CRBM.classes{1,3} = [   [1 0 0 0 1 0 0 1 0]',...
%                          [1 0 0 0 1 0 0 1 0]',...
%                          [1 0 0 0 1 0 0 1 0]']';
%                      
%  CRBM.classes{1,4} = [   [1 0 0 0 1 0 0 0 1]',...
%                          [1 0 0 0 1 0 0 0 1]',...
%                          [1 0 0 0 1 0 0 0 1]']';
%  
%  CRBM.classes{1,5} = [   [1 1 0 0 0 0 0 0 1]',...
%                          [1 1 0 0 0 0 0 0 1]',...
%                          [1 1 0 0 0 0 0 0 1]']';

% %  
%     CRBM.classes{1,5}  = [1 1 0 1 1 1 0]';  %Major 
%     CRBM.classes{1,6}  = [1 0 1 1 1 0 1]';  %Minor
% % 
%    CRBM.classes{1,1} = [[1;0;1],[1;0;1],[1; 0; 1], [1;0;1]]';
%    CRBM.classes{1,2} = [[0;1;0],[0;1;0],[0;1;0],[0;1;0]]';
%    CRBM.classes{1,3} = [[1;1;1],[1;1;1],[1;1;1], [1;1;1]]';
%     CRBM.classes{1,4} = [[0;0;0],[0;0;0],[0;0;0], [0;0;0]]';
%    CRBM.classes{1,5} = [[1;0;1],[1;0;0],[1; 0; 1], [1;0;0]]';
%    CRBM.classes{1,6} = [[0;1;0],[0;0;0],[0;1;0],[0;0;0]]';
%    CRBM.classes{1,7} = [[1;1;1],[0;1;1],[1;1;1], [0;1;1]]';
%     CRBM.classes{1,8} = [[0;0;0],[1;0;0],[0;0;0], [1;0;0]]';

CRBM.data = cell2mat(CRBM.classes');
CRBM.order = 4;
CRBM.numhid = 16 ;
CRBM.numepochs = 2000;
CRBM.numdims = size(CRBM.data,2); %data (visible) dimension
CRBM.model = [];

%% TRAINING

fprintf(1,'Training Layer 1 CRBM , order %d: %d-%d \n',CRBM.order ,CRBM.numdims,CRBM.numhid);
CRBM = train_binarycrbmMikeIdxFix3(CRBM);

%% GENERATING/SAMPLING DATA
% % 
  
classTested =1;

% a = CRBM.classes{1,classTested}(6:7,:);

%  for i = 1:100
    
 numframes = 113; %how many frames to generate
 fr = 1;         %pick a starting frame from initdata                 
% % fprintf(1,'Generating %d-frame sequence of data from %d-layer CRBM ... \n',numframes);

 % classTEST = [[1 0 0 0 0 0 0]', [0 0 1 0 0 0 0]', [1 0 0 0 0 0 0]', [0 0 0 0 1 0 0]']';


GenLog = gen_crbmBinaryMikeVPOSTEST (CRBM, CRBM.classes{1,classTested}(:,:), numframes, fr);

% if a == GenLog.visible(6:7,:)
%     result(i,1) = 1;
% else
%     result(i,1) = 0;
% end
% 
%  end
% % 
% all(result == 1)

%Plot top-layer activations
%  figure(6); 
%  imagesc(GenLog.hidden'); colormap gray;
%  title('First hidden layer, activations '); ylabel('hidden units'); xlabel('frames')
%

% figure(7); 
% imagesc(GenLog.hidden{2}'); 
% colormap gray;title('Second hidden layer, activations'); ylabel('hidden units'); xlabel('frames')
% 