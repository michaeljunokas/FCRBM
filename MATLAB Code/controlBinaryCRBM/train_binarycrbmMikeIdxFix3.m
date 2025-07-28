function [ CRBMConfig ] = train_binarycrbmMikeIdxFix3( CRBMConfig)
% Mike Junokas implementation of training a binary CRBM from code based on 
% Graham Taylor, Geoff Hinton and Sam Roweis work at:

% http://www.cs.nyu.edu/~gwtaylor/publications/nips2006mhmublv/code.html


% This program trains a Conditional Restricted Boltzmann Machine in which
% visible, binary, stochastic inputs are connected to
% hidden, binary, stochastic feature detectors using symmetrically
% weighted connections. Learning is done with 1-step Contrastive Divergence.
% Directed connections are present, from the past order configurations of the
% visible units to the current visible units (A), and the past order
% configurations of the visible units to the current hidden units (B)

% The program assumes that the following variables are set externally:
% CRBMConfig.order          -- order of the model
% CRBMConfig.numepochs      -- maximum number of epochs
% CRBMConfig.numhid         -- number of hidden units 
% CRBMConfig.data           -- a matrix of data (numcases,numdims) 
% CRBMConfig.classes        -- a cell array of the separated classes

% vis = 1;

numdims = size(CRBMConfig.data,2); %visible dimension

epsilonw=1e-2;  %undirected
epsilonbi=1e-2; %visibles
epsilonbj=1e-2; %hidden units
epsilonA=1e-2;  %autoregressive
epsilonB=1e-2;  %prev visibles to hidden

wdecay = 0.0002; %currently we use the same weight decay for w, A, B
mom = 0.9;       %momentum used only after 5 epochs of training
noise = .0001;

%Randomly initialize weights
w = noise*randn(CRBMConfig.numhid,numdims);
bi = noise*randn(numdims,1);
bj = -1+noise*randn(CRBMConfig.numhid,1); %set to favor units being "off"
  
%The autoregressive weights; A(:,:,j) is the weight from t-j to the vis
A = noise*randn(numdims, numdims, CRBMConfig.order);
 
%The weights from previous time-steps to the hiddens; B(:,:,j) is the
%weight from t-j to the hidden layer
B = noise*randn(CRBMConfig.numhid, numdims, CRBMConfig.order);
  
clear wgrad bigrad bjgrad Agrad Bgrad
clear negwgrad negbigrad negbjgrad negAgrad negBgrad
  
%keep previous updates around for momentum
wupdate = zeros(size(w));
biupdate = zeros(size(bi));
bjupdate = zeros(size(bj));
Aupdate = zeros(size(A));
Bupdate = zeros(size(B));

for ii = 1:length(CRBMConfig.classes)
    classLength(ii) = size(CRBMConfig.classes{ii},1);
    if ii == 1
        classIndex = classLength(ii);
    else
        classIndex = [classIndex classIndex(end)+classLength(ii)];
    end
end

for i = 1:length(classIndex)
    range{i} = classIndex(i) - classLength(i) + CRBMConfig.order+1:classIndex(i);
end
INDICES = [range{:}];
    
numcases = length(INDICES);   
data = zeros(numcases,numdims,CRBMConfig.order+1);
data(:,:,1) = CRBMConfig.data(INDICES,:);
    for hh=1:CRBMConfig.order
      data(:,:,hh+1) = CRBMConfig.data(INDICES-hh,:);
    end


% MAIN LOOP
for epoch = 1:CRBMConfig.numepochs,
  errsum=0; %keep a running total of the difference between data and recon    

  %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

    %Calculate contributions from directed autoregressive connections
    bistar = zeros(numdims,numcases); 
    for hh=1:CRBMConfig.order       
      bistar = bistar +  A(:,:,hh)*data(:,:,hh+1)' ;
    end   
    
    %Calculate contributions from directed visible-to-hidden connections
    bjstar = zeros(CRBMConfig.numhid,numcases);
    for hh = 1:CRBMConfig.order
      bjstar = bjstar + B(:,:,hh)*data(:,:,hh+1)';
    end
     
    bottomup = w*data(:,:,1)';
    
    %Calculate "posterior" probability -- hidden state being on 
    %Note that it isn't a true posterior   
    eta =  bottomup + ...                %bottom-up connections
      repmat(bj, 1, numcases) + ...      %static biases on unit
      bjstar;                            %dynamic biases
    
    hposteriors = 1./(1 + exp(-eta));    %logistic
       
    %Activate the hidden units    
    hidstates = hposteriors' > rand(numcases,CRBMConfig.numhid); 
    
    %Calculate positive gradients (note w.r.t. neg energy)
    wgrad = hidstates'*data(:,:,1);
    bigrad = sum(data(:,:,1)' - ...
      repmat(bi,1,numcases) - bistar,2);
    bjgrad = sum(hidstates,1)';
           
    for hh=1:CRBMConfig.order      
      Agrad(:,:,hh) = (data(:,:,1)' -  ...
        repmat(bi,1,numcases) - bistar) * data(:,:,hh+1);
      Bgrad(:,:,hh) = hidstates'*data(:,:,hh+1);      
    end
    
%%%%%%%%% END OF POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
    %Activate the visible units
    topdown = hidstates*w;
    
    eta =  topdown + ...                %top down connections
        repmat(bi',numcases,1) + ...    %static biases
        bistar';                        %dynamic biases
    
% % % % % % % % % %     negdata = 1./(1+exp(-eta));
% % % % % % % % % %     
% % % % % % % % % %     %Now conditional on negdata, calculate "posterior" probability
% % % % % % % % % %     %for hiddens
% % % % % % % % % %     bottomup = w*negdata';
% % % % % % % % % %     eta =  bottomup + ...              %bottom-up connections
% % % % % % % % % %         repmat(bj, 1, numcases) + ...  %static biases on unit (no change)
% % % % % % % % % %         bjstar;                        %dynamic biases (no change)
    
    vposteriors = 1./(1 + exp(-eta));   %logistic
    
    %Calculate negative gradients                                   
%     negwgrad = hposteriors*negdata;
%     negbigrad = sum(negdata' - ...
%         repmat(bi,1,numcases) - bistar,2);
%     negbjgrad = sum(hposteriors,2);
%            
%     for hh=1:CRBMConfig.order      
%       negAgrad(:,:,hh) = (negdata' -  ...
%         repmat(bi,1,numcases) - bistar) * data(:,:,hh+1);
%       negBgrad(:,:,hh) = hposteriors*data(:,:,hh+1);      
%     end
%    
    

% SAYS WE DON'T NEED IT, CAN JUST USE PROBABILITIES DIRECTLY....

%     visstates = double(vposteriors' > rand(numcases,CRBMConfig.numhid));   
  
    %Calculate negative gradients                                   
    negwgrad = hidstates'*vposteriors;
    negbigrad = sum(vposteriors' - ...
        repmat(bi,1,numcases) - bistar,2);
    negbjgrad = sum(hidstates,1)';
           
    for hh=1:CRBMConfig.order      
      negAgrad(:,:,hh) = (vposteriors' -  ...
        repmat(bi,1,numcases) - bistar) * data(:,:,hh+1);
      negBgrad(:,:,hh) = hidstates'*data(:,:,hh+1);      
    end
   
%%%%%%%%% END NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
     err= sum(sum( (data(:,:,1)-vposteriors).^2 ));
     errsum = err + errsum;
    
    if epoch > 5 %use momentum
        momentum=mom;
    else %no momentum
        momentum=0;
    end
    
%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    wupdate =  momentum*wupdate + epsilonw* ...
        ( (wgrad - negwgrad)/numcases - wdecay*w);
    biupdate = momentum*biupdate + ...
        (epsilonbi/numcases)*(bigrad - negbigrad);
    bjupdate = momentum*bjupdate + ...
        (epsilonbj/numcases)*(bjgrad - negbjgrad);

    for hh=1:CRBMConfig.order
        Aupdate(:,:,hh) = momentum*Aupdate(:,:,hh) + ...
            epsilonA* ( (Agrad(:,:,hh) - negAgrad(:,:,hh))/numcases - ...
            wdecay*A(:,:,hh));

        Bupdate(:,:,hh) = momentum*Bupdate(:,:,hh) + ...
            epsilonB* ( (Bgrad(:,:,hh) - negBgrad(:,:,hh))/numcases - ...
            wdecay*B(:,:,hh));
    end

    w = w +  wupdate;
    bi = bi + biupdate;
    bj = bj + bjupdate;

    for hh=1:CRBMConfig.order
        A(:,:,hh) = A(:,:,hh) + Aupdate(:,:,hh);
        B(:,:,hh) = B(:,:,hh) + Bupdate(:,:,hh);
    end
    
%%%%%%%%%%%%%%%% END OF UPDATES  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

  if mod(epoch,100) ==0 
      fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 
  end
end

CRBMConfig.model.w = w;
CRBMConfig.model.bj = bj; 
CRBMConfig.model.bi = bi; 
CRBMConfig.model.A = A; 
CRBMConfig.model.B = B;

end
    
