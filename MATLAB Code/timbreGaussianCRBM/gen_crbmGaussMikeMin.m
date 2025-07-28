function [ GenLog ] = gen_crbmGaussMikeMin ( CRBM , InitData, numframes, fr)

% Mike Junokas implementation of training a binary CRBM from code based on 
%Graham Taylor, Geoff Hinton and Sam Roweis work at:

% http://www.cs.nyu.edu/~gwtaylor/publications/nips2006mhmublv/code.html

%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

% This program uses a single level CRBM (or the first level of a multi-
% level CRBM) to generate data

% The program assumes that the following variables are set externally:
% numframes    -- number of frames to generate
% fr           -- a starting frame from initdata (for initialization)

numGibbs = 30; %number of alternating Gibbs iterations 
numdims = size(InitData,2);
   
%initialize visible layer
visible = zeros(numframes,CRBM.numdims);
visible(1:CRBM.order,:) = InitData(fr:fr+CRBM.order-1,:);

%initialize hidden layer
hidden = ones(numframes,CRBM.numhid);

for tt=CRBM.order+1:numframes
  
  %initialize using the last frame + noise
  visible(tt,:) = visible(tt-1,:) + 0.0001*randn(1,numdims);
  
  %Dynamic biases aren't re-calculated during Alternating Gibbs
  %First, add contributions from autoregressive connections 
  bistar = zeros(numdims,1);
  for hh=1:CRBM.order
    %should modify to data * A'
    bistar = bistar +  CRBM.model.A(:,:,hh)*visible(tt-hh,:)' ;
  end

  %Next, add contributions to hidden units from previous time steps
  bjstar = zeros(CRBM.numhid,1);
  for hh = 1:CRBM.order
    bjstar = bjstar + CRBM.model.B(:,:,hh)*visible(tt-hh,:)';
  end
  
  %Gibbs sampling
  for gg = 1:numGibbs
        % Calculate posterior probability -- hidden state being on (estimate)
        % add in bias
            
        % BINARY ACTIVATION FOR HIDDEN FROM VISIBLE    
        %   p(h_j = 1 | v) = ?(b_j + ?(v_i * w_ij))

        % BOTTOMUP = LEARNED WEIGHT/CONNECTION BETWEEN VISIBLE AND HIDDEN UNITS * A
        % GIVEN VISIBLE VECTOR... ?(v_i * w_ij)
            bottomup =  CRBM.model.w*(visible(tt,:))';                              

        % ETA = THE ABOVE BOTTOM UP WITH BIASES...b_j + ?(v_i * w_ij)   
            eta = (bottomup./CRBM.gsd) + ...                   %bottom-up connections
              CRBM.model.bj + ...                  %static biases on unit
              bjstar;                              %dynamic biases
        
        % HPOSTERIORS = SIGMOID OF ABOVE ETA...?(b_j + ?(v_i * w_ij))  
            hposteriors = 1./(1 + exp(-eta));      %logistic
       
        % IS THE UNIT ACTIVATED OR NOT? STOCHASTICALLY COMPARED TO SIGMOID

  hidden(tt,:) = double(hposteriors' > rand(1,CRBM.numhid));

   %DOWNWARD PASS

        % BINARY ACTIVATION FOR VISIBLE FROM HIDDEN    
        %   p(v_i = 1 | h) = ?(a_i + ?(h_j * w_ij))

        % TOPDOWN = LEARNED WEIGHT/CONNECTION BETWEEN HIDDEN AND VISIBLE UNITS
        % (SAME AS ABOVE VISIBLE AND HIDDEN) * A GIVEN HIDDEN VECTOR... ?(h_j * w_ij)
        % Visibles are Gaussian units, so find the mean of the Gaussian...   
        %
        topdown = CRBM.gsd.*(hidden(tt,:)*CRBM.model.w);

        % VISIBLE = THE ABOVE TOPDOWN WITH BIASES...a_i + ?(h_j * w_ij)
        %... then do a mean-field approximation
        
            visible(tt,:) = topdown + ...            %top down connections
              CRBM.model.bi' + ...                   %static biases
              bistar';                               %dynamic biases   

   end

  %If we are done Gibbs sampling, then do a mean-field sample
  %(otherwise very noisy)                        trial 1
  topdown = CRBM.gsd.*(hposteriors'*CRBM.model.w);                 

    visible(tt,:) = topdown + ...                        %top down connections
      CRBM.model.bi' + ...                             %static biases
      bistar';  
  
 
end

GenLog.visible = visible;
GenLog.hidden = hidden;

end

  

