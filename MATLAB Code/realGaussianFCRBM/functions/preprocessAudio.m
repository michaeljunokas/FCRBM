% Version 0.100 (Unsupported, unreleased) 
%
% Code provided by Graham Taylor and Geoff Hinton 
%
% For more information, see:
%     http://www.cs.toronto.edu/~gwtaylor/publications/nips2006mhmublv
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
% We support two types of skeletons:
%  1) Those built from the CMU database (acclaim)
%     http://mocap.cs.cmu.edu/
%  2) Those built from data from Eugene Hsu (mit)
%     http://people.csail.mit.edu/ehsu/work/sig05stf/
% This program preprocesses data (stage 2 of 2)
% We extract the dimensions of interest and form "mini-batches"
% We also scale the data.
% Certain joint angles are 1-2 DOF, so don't model constant zero cols
%

clear batchdata minibatchdata batchdataindex

n1=nt;
batchsize = 100;        %size of minibatches
numvalidatebatches = 0; %num of minibatches to use for the validation set

%combine the data into a large batch
batchdata = cell2mat(Audio'); %flatten it into a standard 2d array
% batchdata = batchdata(:,indx);
numcases = size(batchdata,1);

%Normalize the data
data_mean = mean(batchdata,1);
data_std = std(batchdata);
batchdata =( batchdata - repmat(data_mean,numcases,1) ) ./ ...
  repmat( data_std, numcases,1);

%Index the valid cases (we don't want to mix sequences)
%This depends on the order of our model
for jj=1:length(Audio)
  seqlengths(jj) = size(Audio{jj},1);
  if jj==1 %first sequence
    batchdataindex = n1+1:seqlengths(jj);
  else
    batchdataindex = [batchdataindex batchdataindex(end)+n1+1: ...
      batchdataindex(end)+seqlengths(jj)];
  end
end

% Here's a convenient place to remove offending frames from the index
% example: offendingframes = [231 350 714 1121];
% batchdataindex = setdiff(batchdataindex,offendingframes);


%now that we know all the valid starting frames, we can randomly permute
%the order, such that we have a balanced training set
permindex = batchdataindex(randperm(length(batchdataindex)));

%should we keep some minibatches as a validation set?
numfullbatches = floor(length(permindex)/batchsize);

%fit all minibatches of size batchsize
%note that since reshape works in colums, we need to do a transpose here
minibatchindex = reshape(permindex(1: ...
  batchsize*(numfullbatches-numvalidatebatches)),...
  batchsize,numfullbatches-numvalidatebatches)';
%no need to reshape the validation set into mini-batches
%we treat it as one big batch
validatebatchindex = permindex(...
  batchsize*(numfullbatches-numvalidatebatches)+1: ...
  batchsize*numfullbatches);  
%Not all minibatches will be the same length ... must use a cell array (the
%last batch is different)
minibatch = num2cell(minibatchindex,2);
%tack on the leftover frames (smaller last batch)
leftover = permindex(batchsize*numfullbatches+1:end);
minibatch = [minibatch;num2cell(leftover,2)];

