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
% This program trains a model that is identical to a standard crbm with
% Gaussian visible units but the weights are factored no three-way
% interactions does not use any label data

%batchdata is a big matrix of all the frames
%we index it with "minibatch", a cell array of mini-batch indices
 %visible dimension

errRate = cell(1,2);
errRate{1,1} = 0;
errRate{1,2} = 0;
erRate = 1;
%Setting learning rates
%Corresponding to the "undirected" observation model
epsilonvisfac=single(1e-6);
epsilonhidfac=single(1e-6);

%Corresponding to the "directed" Autoregressive model
epsilonpastfacA=single(1e-7);
epsilonvisfacA=single(1e-7);

%Corresponding to the "directed" past->hidden model
epsilonpastfacB=single(1e-6);
epsilonhidfacB=single(1e-6);

epsilonvisbias=single(1e-6);
epsilonhidbias=single(1e-6);

%currently we use the same weight decay for all weights
%but no weight decay for biases
wdecay = single(0.0002);

mom = single(0.9);       %momentum used only after 5 epochs of training

if restart==1,  
  restart=0;
  epoch=1;
 
  %weights  
  visfac = single(0.01*randn(numdims,numfac));  
  hidfac = single(0.01*randn(numhid,numfac));
    
  %Note the new parameterization of pastfac:
  %First numdims rows correspond to time t-nt
  %Last numdims rows correspond to time t-1
  pastfacA = single(0.01*randn(nt*numdims,numfac));   
  visfacA = single(0.01*randn(numdims,numfac));
  
  pastfacB = single(0.01*randn(nt*numdims,numfac));  
  hidfacB = single(0.01*randn(numhid,numfac));      
    
  %biases
  visbiases = zeros(1,numdims,'single');
  hidbiases = zeros(1,numhid,'single');
  %vishid = 0.01*randn(numdims,numhid);    

  %keep previous updates around for momentum
  visfacinc = zeros(size(visfac),'single');  
  hidfacinc = zeros(size(hidfac),'single');
  
  pastfacAinc = zeros(size(pastfacA),'single');
  visfacAinc = zeros(size(visfacA),'single');  
  
  pastfacBinc = zeros(size(pastfacB),'single');
  hidfacBinc = zeros(size(hidfacB),'single');  
     
  visbiasinc = zeros(size(visbiases),'single');
  hidbiasinc = zeros(size(hidbiases),'single');
  %vishidinc = zeros(size(vishid));    
end


%Main loop
while erRate >= 1

%    for epoch = epoch:maxepoch,
      errsum=0; %keep a running total of the difference between data and reco   

    %%%%%%%%% START POSITIVE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %calculate inputs to factors (will be used many times)
        yvis = data*visfac; %summing over numdims

        ypastA = past*pastfacA;     %summing over nt*numdims
        yvisA = data*visfacA;       %summing over numdims

        ypastB = past*pastfacB;     %summing over nt*numdims            


        %pass 3-way term + gated biases + hidbiases through sigmoid 
        poshidprobs = 1./(1 + exp(-yvis*hidfac'  ...
          -ypastB*hidfacB' - repmat(hidbiases,numcases,1)));
          %-data*vishid - repmat(hidbiases,numcases,1)));  

        %Activate the hidden units    
        hidstates = single(poshidprobs > rand(numcases,numhid));

        yhid = hidstates*hidfac;
        yhid_ = poshidprobs*hidfac; %smoothed version

        yhidB_ = poshidprobs*hidfacB; %smoothed version  

        %these are used multiple times, so cache
        %yvishid_ = yvis.*yhid_;
        %yvispastA = yvisA.*ypastA;
        %ypasthidB_ = ypastB.*yhidB_;


        %Calculate statistics needed for gradient update
        %Gradients are taken w.r.t neg energy
        %Note that terms that are common to positive and negative stats
        %are left out
        posvisprod = data'*yhid_; %smoothed
        poshidprod = poshidprobs'*yvis; %smoothed

        posvisAprod = data'*ypastA;
        pospastAprod =  past'*yvisA;

        pospastBprod = past'*yhidB_; %smoothed    
        poshidBprod =  poshidprobs'*ypastB;

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
        negdata = yhid*visfac' + ...
          ypastA*visfacA' + ...
          repmat(visbiases,numcases,1);    

        yvis = negdata*visfac;   

        %pass 3-way term + gated biases + hidbiases through sigmoid 
        neghidprobs = 1./(1 + exp(-yvis*hidfac'  ...
          -ypastB*hidfacB' - repmat(hidbiases,numcases,1)));

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
          %yvishid_ = yvis.*yhid_;
          yvisA = negdata*visfacA;       %summing over numdims
          %yvispastA = yvisA.*ypastA;
          %ypasthidB_ = ypastB.*yhidB_;

          %last cd step -- Calculate statistics needed for gradient update
          %Gradients are taken w.r.t neg energy
          %Note that terms that are common to positive and negative stats
          %are left out
          negvisprod = negdata'*yhid_; %smoothed
          neghidprod = neghidprobs'*yvis; %smoothed

          negvisAprod = negdata'*ypastA;
          negpastAprod =  past'*yvisA;

          negpastBprod = past'*yhidB_; %smoothed
          neghidBprod =  neghidprobs'*ypastB;

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
    hidfacinc = momentum*hidfacinc + ...
      epsilonhidfac*( (poshidprod - neghidprod)/numcases - wdecay*hidfac);

    visfacAinc = momentum*visfacAinc + ...
      epsilonvisfacA*( (posvisAprod - negvisAprod)/numcases - wdecay*visfacA);
    pastfacAinc = momentum*pastfacAinc + ...
      epsilonpastfacA*( (pospastAprod - negpastAprod)/numcases - wdecay*pastfacA);

    hidfacBinc = momentum*hidfacBinc + ...
      epsilonhidfacB*( (poshidBprod - neghidBprod)/numcases - wdecay*hidfacB);
    pastfacBinc = momentum*pastfacBinc + ...
      epsilonpastfacB*( (pospastBprod - negpastBprod)/numcases - wdecay*pastfacB);

    visbiasinc = momentum*visbiasinc + ...
      (epsilonvisbias/numcases)*(posvisact - negvisact);
    hidbiasinc = momentum*hidbiasinc + ...
      (epsilonhidbias/numcases)*(poshidact - neghidact);

    visfac = visfac + visfacinc;
    hidfac = hidfac + hidfacinc;

    visfacA = visfacA + visfacAinc;
    pastfacA = pastfacA + pastfacAinc;

    hidfacB = hidfacB + hidfacBinc;
    pastfacB = pastfacB + pastfacBinc;

    visbiases = visbiases + visbiasinc;
    hidbiases = hidbiases + hidbiasinc;

    %%%%%%%%%%%%%%%% END OF UPDATES  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      %every 10 epochs, show output
      if mod(epoch,2) ==0
          fprintf(1, 'epoch %4i error %6.1f rate %6.1f  \n', epoch, errsum, erRate);
      end
      
      epoch = epoch+1;
      
      %   end

end