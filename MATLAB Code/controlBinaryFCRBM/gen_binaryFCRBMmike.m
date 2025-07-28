function [ GenLog ] = gen_binaryFCRBMmike ( FCRBM , InitData, labels, numframes, fr)

% Mike Junokas implementation of training a binary CRBM from code based on 
%Graham Taylor, Geoff Hinton and Sam Roweis work at:

% http://www.cs.nyu.edu/~gwtaylor/publications/nips2006mhmublv/code.html

% The program assumes that the following variables are set externally:
% numframes    -- number of frames to generate
% fr           -- a starting frame from initdata (for initialization)

numGibbs = 30; %number of alternating Gibbs iterations 
numdims = size(InitData,2);
numhid = FCRBM.numhid;
%initNoise = 0.05;

%initialize visible layer
visible = zeros(numframes,FCRBM.numdims);
visible(1:FCRBM.order,:) = InitData(fr:fr+FCRBM.order-1,:);

%initialize hidden layer
poshidprobs = zeros(numframes,numhid);
hidstates = ones(numframes,numhid); %"hidden" in other CRBM
past = zeros(1,FCRBM.order*numdims);

%unpacking FCRBM struct (for clarity, can be removed to streamline later

visfac=FCRBM.model.visfac;
featfac=FCRBM.model.featfac;
hidfac=FCRBM.model.hidfac;
visfacA=FCRBM.model.visfacA;
pastfacA=FCRBM.model.pastfacA;
hidfacB=FCRBM.model.hidfacB;
pastfacB=FCRBM.model.pastfacB;
labelfeat=FCRBM.model.labelfeat;
visbiases=FCRBM.model.visbiases;
hidbiases=FCRBM.model.hidbiases;

for tt=FCRBM.order+1:numframes
  
  %initialize using the last frame + noise
    visible(tt,:) = visible(tt-1,:); %%%%% +initNoise*randn(1,numdims); NO NOISE FOR BINARY UNITS
  
  %Dynamic biases aren't re-calculated during Alternating Gibbs
  %First, add contributions from autoregressive connections 
 
    for hh=FCRBM.order:-1:1 %note reverse order
     past(:,numdims*(FCRBM.order-hh)+1:numdims*(FCRBM.order-hh+1)) = visible(tt-hh,:);
    %Cheat and use initdata instead of generated data
    %Note the conversion to the absolute fr scale instead of the relative
    %tt scale
    %past(:,numdims*(nt-hh)+1:numdims*(nt-hh+1)) = initdata(fr+tt-hh-1,:);
    end 
    
  %Input from features & past does not change during Alternating Gibbs 
  %Set these now and leave them
  features = labels(tt,:)*labelfeat;
  
  %undirected model
  yfeat = features*featfac;

  %autoregressive model
  ypastA = past*pastfacA;
  yfeatA = features*featfac;
  
  %directed vis-hid model
  ypastB = past*pastfacB;
  yfeatB = features*featfac;

  %constant term during inference
  %(not dependent on visibles)
  constinf = -(ypastB.*yfeatB)*hidfacB' - hidbiases;
  
  %constant term during reconstruction
  %(not dependent on hiddens)
  constrecon = (yfeatA.*ypastA)*visfacA' + visbiases;
  
  
  
  for gg = 1:numGibbs
    
    yvis = visible(tt,:)*visfac;    
   %pass through sigmoid    
    %only part from "undirected" model changes
    poshidprobs(tt,:) = 1./(1 + exp(-(yvis.*yfeat)*hidfac' + constinf));     

    %Activate the hidden units
    hidstates(tt,:) = single(poshidprobs(tt,:) > rand(1,numhid));

    yhid = hidstates(tt,:)*hidfac;
      
      %NEGATIVE PHASE
    %Don't add noise at visibles
    %Note only the "undirected" term changes
    vposteriors = 1./(1 + exp(-(yfeat.*yhid)*visfac' + constrecon));   
   
% % % %         % VPOSTERIORS = SIGMOID OF ABOVE ETA...?(a_i + ?(h_j * w_ij))
% % % %              vposteriors = 1./(1 + exp(-eta));      %logistic
% % % %             
% % % %              % IS THE UNIT ACTIVATED OR NOT? STOCHASTICALLY COMPARED TO SIGMOID
% % % %  
   visible(tt,:) = double(vposteriors >rand(1,numdims));

   end

  %Now do mean-field
  yhid_ = poshidprobs(tt,:)*hidfac; %smoothed version

  %Mean-field approx
 vposteriors = 1./(1 + exp(-(yfeat.*yhid)*visfac' + constrecon));   
  visible(tt,:) = double(vposteriors >rand(1,numdims));

  
end

GenLog.visible = visible;
GenLog.hidden = hidstates;

end

  

