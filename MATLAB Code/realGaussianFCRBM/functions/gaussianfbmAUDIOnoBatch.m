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
% Train a factored, conditional RBM which has label units that modulate
% each pair of interactions
% CRBM has gaussian visible and binary stochastic hidden units
% Standard dev on Gaussian units is fixed to 1
% No weight-sharing
%
% The program assumes that the following variables are set externally:
% nt        -- order of the model
% numepochs -- maximum number of epochs
% numhid    -- number of hidden units 
% numfeat   -- number of real-valued features between labels and factors 
% numfac    --  number of factors
% batchdata --  a matrix of data (numcases,numdims) 
% minibatch -- a cell array of dimension batchsize, indexing the valid
% frames in batchdata
% restart   -- set to 1 if learning starts from beginning 

%batchdata is a big matrix of all the frames
%we index it with "minibatch", a cell array of mini-batch indices


errRate = cell(1,2);
errRate{1,1} = 0;
errRate{1,2} = 0;
erRate = 1;

%Setting learning rates
%Corresponding to the "undirected" observation model
epsilonvisfac=single(1e-6);
epsilonfeatfac=single(1e-6);
epsilonhidfac=single(1e-6);

%Corresponding to the "directed" Autoregressive model
epsilonpastfacA=single(1e-6);
epsilonfeatfacA=single(1e-6);
epsilonvisfacA=single(1e-6);

%Corresponding to the "directed" past->hidden model
epsilonpastfacB=single(1e-6);
epsilonfeatfacB=single(1e-6);
epsilonhidfacB=single(1e-6);

epsilonlabelfeat=single(1e-6);

epsilonvisbias=single(1e-6);
epsilonhidbias=single(1e-6);
%epsilonvishid=1e-3;  %gated biases

%currently we use the same weight decay for all weights
%but no weight decay for biases
wdecay = single(0.0002);

mom = single(0.9);       %momentum used only after 5 epochs of training

if restart==1,  
  restart=0;
  epoch=1;
 
  %weights  
  visfac = single(0.01*randn(numdims,numfac));
  featfac = single(0.01*randn(numfeat,numfac));
  hidfac = single(0.01*randn(numhid,numfac));
    
  %Note the new parameterization of pastfac:
  %First numdims rows correspond to time t-nt
  %Last numdims rows correspond to time t-1
  pastfacA = single(0.01*randn(nt*numdims,numfac)); 
  featfacA = single(0.01*randn(numfeat,numfac));
  visfacA = single(0.01*randn(numdims,numfac));
  
  pastfacB = single(0.01*randn(nt*numdims,numfac));
  featfacB = single(0.01*randn(numfeat,numfac));
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
  featfacAinc = zeros(size(featfacA),'single');
  visfacAinc = zeros(size(visfacA),'single');  
  
  pastfacBinc = zeros(size(pastfacB),'single');
  featfacBinc = zeros(size(featfacB),'single');
  hidfacBinc = zeros(size(hidfacB),'single');  
  
  labelfeatinc = zeros(size(labelfeat),'single');
  
  visbiasinc = zeros(size(visbiases),'single');
  hidbiasinc = zeros(size(hidbiases),'single');
  %vishidinc = zeros(size(vishid));    
end



%Main loop
while erRate >= 1
  errsum=0; %keep a running total of the difference between data and recon
  

%%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    %get the features from the one-hot labels

    features = labels*labelfeat;

    %DEBUG
    %past = double(rand(size(past))>0.5);
    %calculate inputs to factors (will be used many times)
    yvis = data*visfac; %summing over numdims
    yfeat = features*featfac; %summing over numfeat
        
    ypastA = past*pastfacA;     %summing over nt*numdims
    yfeatA = features*featfacA; %summing over numfeat
    yvisA = data*visfacA;       %summing over numdims
    
    ypastB = past*pastfacB;     %summing over nt*numdims
    yfeatB = features*featfacB; %summing over numfeat
        
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
    posfeatgrad = (yvishid_)*featfac' + ...
      (yvispastA)*featfacA' + ...
      (ypasthidB_)*featfacB'; 
           
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

       errRate{1,2} = errRate{1,1};
       errRate{1,1} = errsum;
       erRate = abs(errRate{1,1} - errRate{1,2});     
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
      negfeatgrad = (yvishid_)*featfac' + ...
        (yvispastA)*featfacA' + ...
        (ypasthidB_)*featfacB';

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
  epsilonfeatfac*( (posfeatprod - negfeatprod)/numcases - wdecay*featfac);
hidfacinc = momentum*hidfacinc + ...
  epsilonhidfac*( (poshidprod - neghidprod)/numcases - wdecay*hidfac);

visfacAinc = momentum*visfacAinc + ...
  epsilonvisfacA*( (posvisAprod - negvisAprod)/numcases - wdecay*visfacA);
featfacAinc = momentum*featfacAinc + ...
  epsilonfeatfacA*( (posfeatAprod - negfeatAprod)/numcases - wdecay*featfacA);
pastfacAinc = momentum*pastfacAinc + ...
  epsilonpastfacA*( (pospastAprod - negpastAprod)/numcases - wdecay*pastfacA);

hidfacBinc = momentum*hidfacBinc + ...
  epsilonhidfacB*( (poshidBprod - neghidBprod)/numcases - wdecay*hidfacB);
featfacBinc = momentum*featfacBinc + ...
  epsilonfeatfacB*( (posfeatBprod - negfeatBprod)/numcases - wdecay*featfacB);
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
featfacA = featfacA + featfacAinc;
pastfacA = pastfacA + pastfacAinc;

hidfacB = hidfacB + hidfacBinc;
featfacB = featfacB + featfacBinc;
pastfacB = pastfacB + pastfacBinc;

labelfeat = labelfeat + labelfeatinc;


visbiases = visbiases + visbiasinc;
hidbiases = hidbiases + hidbiasinc;
    
%%%%%%%%%%%%%%%% END OF UPDATES  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if mod(epoch,2) ==0
      fprintf(1, 'epoch %4i error %6.1f rate %6.1f  \n', epoch, errsum, erRate);  
    end
    epoch = epoch+1;
      
end

