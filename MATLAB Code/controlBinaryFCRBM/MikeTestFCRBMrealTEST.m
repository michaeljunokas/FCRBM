function [ FCRBMConfig ] = MikeTestFCRBMrealTEST(FCRBMConfig)
% Mike Junokas implementation of training a binary CRBM from code based on 
% Graham Taylor, Geoff Hinton and Sam Roweis work at:

% http://www.cs.nyu.edu/~gwtaylor/publications/nips2006mhmublv/code.html

% on gaussianfbm.m

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

numdims = size(FCRBMConfig.data,2); %visible dimension
numlabels = size(FCRBMConfig.labels,2);
numfac = FCRBMConfig.numfac;
numfeat = FCRBMConfig.numfeat;
numhid = FCRBMConfig.numhid;
order = FCRBMConfig.order;
numepochs = FCRBMConfig.numepochs;
cdsteps = FCRBMConfig.cdSteps;
data = FCRBMConfig.data;
labels = FCRBMConfig.labels;

%Setting learning rates
%Corresponding to the "undirected" observation model
epsilonvisfac=single(1e-2);
%only one set of featfac parameters
%shared between undirected, A & B models
epsilonfeatfac=single(1e-2);
epsilonhidfac=single(1e-2);

%Corresponding to the "directed" Autoregressive model
epsilonpastfacA=single(1e-3);
epsilonvisfacA=single(1e-3);

%Corresponding to the "directed" past->hidden model
epsilonpastfacB=single(1e-2);
epsilonhidfacB=single(1e-2);

epsilonlabelfeat=single(1e-3);

epsilonvisbias=single(1e-2);
epsilonhidbias=single(1e-2);
%epsilonvishid=1e-3;  %gated biases

%currently we use the same weight decay for all weights
%but no weight decay for biases
wdecay = single(0.0002);

mom = single(0.9);       %momentum used only after 5 epochs of training
 
  %weights  
  visfac = single(0.01*randn(numdims,numfac));
  featfac = single(0.01*randn(numfeat,numfac));
  hidfac = single(0.01*randn(numhid,numfac));
    
  %Note the new parameterization of pastfac:
  %First numdims rows correspond to time t-nt
  %Last numdims rows correspond to time t-1
  pastfacA = single(0.01*randn(order*numdims,numfac)); 
  visfacA = single(0.01*randn(numdims,numfac));
  
  pastfacB = single(0.01*randn(order*numdims,numfac));
  hidfacB = single(0.01*randn(numhid,numfac));
      
  %matrix where rows are per-label features
  labelfeat = single(0.01*randn(numlabels,numfeat));  
  
  %biases
  visbiases = zeros(1,numdims,'single');
  hidbiases = zeros(1,numhid,'single');
  %vishid = 0.01*randn(numdims,numhid);
     
  clear posdataprod pospastprod poshidprod posvishidprod posvisact poshidact
  clear negdataprod negpastprod neghidprod negvishidprod negvisact neghidact

  %keep previous updates around for momentum
  visfacinc = zeros(size(visfac),'single');
  featfacinc = zeros(size(featfac),'single');
  hidfacinc = zeros(size(hidfac),'single');
 
  pastfacAinc = zeros(size(pastfacA),'single');
  visfacAinc = zeros(size(visfacA),'single');  
  
  pastfacBinc = zeros(size(pastfacB),'single');
  hidfacBinc = zeros(size(hidfacB),'single');  
  
  labelfeatinc = zeros(size(labelfeat),'single');
  
  visbiasinc = zeros(size(visbiases),'single');
  hidbiasinc = zeros(size(hidbiases),'single');
  %vishidinc = zeros(size(vishid));    



for epoch = 1:numepochs,
  errsum=0; %keep a running total of the difference between data and recon    

  %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    numcases = size(FCRBMConfig.data,1); %%was cases
    past = zeros(numcases,order*numdims,1);   
    
        for hh=order:-1:1    
            past(:,numdims*(order-hh)+1:numdims*(order-hh+1)) = data(:,:); % + randn(numcases,numdims) No noise in binary
        end  

    %get the features from the one-hot labels
%     labels = FCRBMConfig.labels;  % matching somehow?
    features = labels*labelfeat;
    
    
      %DEBUG
    %past = double(rand(size(past))>0.5);
    %calculate inputs to factors (will be used many times)
    yvis = data*visfac; %summing over numdims
    yfeat = features*featfac; %summing over numfeat
        
    ypastA = past*pastfacA;     %summing over nt*numdims
    yfeatA = features*featfac;  %summing over numfeat
    yvisA = data*visfacA;       %summing over numdims
    
    ypastB = past*pastfacB;     %summing over nt*numdims
    yfeatB = features*featfac;  %summing over numfeat
        
    yvisfeat = yvis.*yfeat; %used twice, so cache
    ypastfeatB = ypastB.*yfeatB; %used twice, so cache
    
    %pass 3-way term + gated biases + hidbiases through sigmoid 
    poshidprobs = 1./(1 + exp(-yvisfeat*hidfac'  ...
      -ypastfeatB*hidfacB' - repmat(hidbiases,numcases,1)));
      %-data*vishid - repmat(hidbiases,numcases,1)));  
    
    %Activate the hidden units    
    hidstates = single(poshidprobs > rand(numcases,numhid));
    
    yhid = hidstates*hidfac;
    yhid_ = poshidprobs*hidfac; %smoothed version
    
    yhidB_ = poshidprobs*hidfacB; %smoothed version  
    
    %these are used multiple times, so cache
    yvishid_ = yvis.*yhid_;
    yvispastA = yvisA.*ypastA;
    ypasthidB_ = ypastB.*yhidB_;
    yfeatpastA = yfeatA.*ypastA;                    
    
    %Calculate statistics needed for gradient update
    %Gradients are taken w.r.t neg energy
    %Note that terms that are common to positive and negative stats
    %are left out
    posvisprod = data'*(yfeat.*yhid_); %smoothed
    posfeatprod = features'*(yvishid_); %smoothed
    poshidprod = poshidprobs'*(yvisfeat); %smoothed
    
    posvisAprod = data'*(yfeatpastA);
    posfeatAprod = features'*(yvispastA);
    pospastAprod =  past'*(yvisA.*yfeatA);
   
    pospastBprod = past'*(yfeatB.*yhidB_); %smoothed
    posfeatBprod =  features'*(ypasthidB_); %smoothed
    poshidBprod =  poshidprobs'*(ypastfeatB);
    
    %Now the gradients for the label/feature matrix
    %First find the grad terms w.r.t. the features
    %Then backpropagate (it's linear, so simply matrix multiply)
    %There are three terms, since the features gate the undirected & two
    %sets of directed connections
   posfeatgrad = (yvishid_ + yvispastA + ypasthidB_)*featfac';
           
    %posvishidprod = data'*poshidprobs;
    posvisact = sum(data,1);
    poshidact = sum(poshidprobs,1);  %smoothed                          
    
%%%%%%%%% END OF POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%% START NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

  for cdn = 1:cdsteps    
    %Activate the visible units
    %Collect 3-way terms + vis biases + gated biases 
    %note use of stochastic hidstates
    %Mean-field version (do not add Gaussian noise)        
    negdata = (yfeat.*yhid)*visfac' + ...
      (yfeatpastA)*visfacA' + ...
      repmat(visbiases,numcases,1);    
    
    yvis = negdata*visfac;
    yvisfeat = yvis.*yfeat; %used twice, so cache     
    
    %pass 3-way term + gated biases + hidbiases through sigmoid 
    neghidprobs = 1./(1 + exp(-yvisfeat*hidfac'  ...
      -ypastfeatB*hidfacB' - repmat(hidbiases,numcases,1)));

    if cdn == 1
      %Calculate reconstruction error
      err= sum(sum( (data(:,:,1)-negdata).^2 ));
      errsum = err + errsum;
    end
 
    if cdn == cdsteps     
      yhidB_ = neghidprobs*hidfacB; %smoothed version 
      yhid_ = neghidprobs*hidfac; %smoothed version
      yvishid_ = yvis.*yhid_;
      yvisA = negdata*visfacA;       %summing over numdims
      yvispastA = yvisA.*ypastA;
      ypasthidB_ = ypastB.*yhidB_;
      %last cd step -- Calculate statistics needed for gradient update
      %Gradients are taken w.r.t neg energy
      %Note that terms that are common to positive and negative stats
      %are left out
      negvisprod = negdata'*(yfeat.*yhid_); %smoothed
      negfeatprod = features'*(yvishid_); %smoothed
      neghidprod = neghidprobs'*(yvisfeat); %smoothed

      negvisAprod = negdata'*(yfeatpastA);
      negfeatAprod = features'*(yvispastA);
      negpastAprod =  past'*(yvisA.*yfeatA);

      negpastBprod = past'*(yfeatB.*yhidB_); %smoothed
      negfeatBprod =  features'*(ypasthidB_); %smoothed
      neghidBprod =  neghidprobs'*(ypastfeatB);

      %Now the gradients for the label/feature matrix
      %First find the grad terms w.r.t. the features
      %Then backpropagate (it's linear, so simply matrix multiply)
      %There are three terms, since the features gate the undirected & two
      %sets of directed connections
      negfeatgrad = (yvishid_ + yvispastA + ypasthidB_)*featfac';
      
      %negvishidprod = data'*neghidprobs;
      negvisact = sum(negdata,1);
      neghidact = sum(neghidprobs,1);  %smoothed

    else
      %Stochastically sample the hidden units
      hidstates = single(neghidprobs > rand(numcases,numhid));      
      yhid = hidstates*hidfac;
    end 
  end
      
     

%%%%%%%%% END NEGATIVE PHASE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
    
    if epoch > 5 %use momentum
        momentum=mom;
    else %no momentum
        momentum=0;
    end
    
%%%%%%%%% UPDATE WEIGHTS AND BIASES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
visfacinc = momentum*visfacinc + ...
  epsilonvisfac*( ( posvisprod - negvisprod)/numcases - wdecay*visfac);

featfacinc = momentum*featfacinc + ...
  epsilonfeatfac*((posfeatprod + posfeatAprod + posfeatBprod ...
  - negfeatprod - negfeatAprod - negfeatBprod)/numcases - wdecay*featfac);
% featfacinc = momentum*featfacinc + ...
%   epsilonfeatfac*( (posfeatprod - negfeatprod)/numcases - wdecay*featfac);

hidfacinc = momentum*hidfacinc + ...
  epsilonhidfac*( (poshidprod - neghidprod)/numcases - wdecay*hidfac);

visfacAinc = momentum*visfacAinc + ...
  epsilonvisfacA*( (posvisAprod - negvisAprod)/numcases - wdecay*visfacA);
% featfacAinc = momentum*featfacAinc + ...
%   epsilonfeatfacA*( (posfeatAprod - negfeatAprod)/numcases - wdecay*featfacA);
pastfacAinc = momentum*pastfacAinc + ...
  epsilonpastfacA*( (pospastAprod - negpastAprod)/numcases - wdecay*pastfacA);

hidfacBinc = momentum*hidfacBinc + ...
  epsilonhidfacB*( (poshidBprod - neghidBprod)/numcases - wdecay*hidfacB);
% featfacBinc = momentum*featfacBinc + ...
%   epsilonfeatfacB*( (posfeatBprod - negfeatBprod)/numcases - wdecay*featfacB);
pastfacBinc = momentum*pastfacBinc + ...
  epsilonpastfacB*( (pospastBprod - negpastBprod)/numcases - wdecay*pastfacB);

labelfeatinc = momentum*labelfeatinc + ...
  epsilonlabelfeat*( labels'*(posfeatgrad - negfeatgrad)/numcases - wdecay*labelfeat);

visbiasinc = momentum*visbiasinc + ...
  (epsilonvisbias/numcases)*(posvisact - negvisact);
hidbiasinc = momentum*hidbiasinc + ...
  (epsilonhidbias/numcases)*(poshidact - neghidact);


visfac = visfac + visfacinc;
featfac = featfac + featfacinc;
hidfac = hidfac + hidfacinc;

visfacA = visfacA + visfacAinc;
pastfacA = pastfacA + pastfacAinc;

hidfacB = hidfacB + hidfacBinc;
pastfacB = pastfacB + pastfacBinc;

labelfeat = labelfeat + labelfeatinc;

%sfigure(34); imagesc(labelfeat); colormap gray; axis off     
%drawnow;

visbiases = visbiases + visbiasinc;
hidbiases = hidbiases + hidbiasinc;
  %every 10 epochs, show output
  if mod(epoch,10) ==0
      fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum);
  end
  
%%%%%%%%%%%%%%%% END OF UPDATES  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


      
FCRBMConfig.model.visfac = visfac;
FCRBMConfig.model.featfac = featfac; 
FCRBMConfig.model.hidfac = hidfac; 
FCRBMConfig.model.visfacA = visfacA; 
FCRBMConfig.model.pastfacA = pastfacA;
FCRBMConfig.model.hidfacB = hidfacB; 
FCRBMConfig.model.pastfacB = pastfacB;
FCRBMConfig.model.labelfeat = labelfeat;
FCRBMConfig.model.visbiases = visbiases;
FCRBMConfig.model.hidbiases = hidbiases;

end
