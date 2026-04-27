%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% AUTOREGRESSIVE PROJECT CHECKPOINT
%%% Demonstration of autoregressive (AR) models up to second order.
%%% Colorado School of Mines EENG311 - Spring 2026 - Mike Wakin
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% In contrast to the Hawaiian text dataset, some data sequences have more
% of an algebraic structure over time. 
%
% To get a sense of this, let's create our own sequence that has a
% particular "formula" for creating each symbol based on the few most 
% recent symbols that preceded it: 
% 
% sym(n) ~= 0.8*sym(n-1) - 0.2*sym(n-2) + 0.4*sym(n-3)
% 
% We also add a bit of noise for randomness, and make sure each symbol is
% an integer between 1 and 9.
%
% This type of model is known as an Autoregressive (AR) model.
ARcoeffs = [0.8 -0.2 0.4]'; 
ARconst = 0;
ARorder = length(ARcoeffs);
NoiseVar = 0.15; % this is the variance of the additive Gaussian noise
sequenceLength = 1000;
sequence = zeros(sequenceLength,1);
sequence(1:ARorder) = ceil(9*rand(ARorder,1));
for ii = ARorder+1:sequenceLength
    nextSym = sum(ARcoeffs.*sequence(ii-1:-1:ii-ARorder)) + ARconst + sqrt(NoiseVar)*randn;
    nextSym = round(nextSym);
    nextSym = min(nextSym,9);
    nextSym = max(nextSym,1);
    sequence(ii) = nextSym;
end
figure(1);clf;plot(sequence,'o');

% Suppose we want to train a forecasting model for this data. We could
% build a Markov Chain model, but it could require a pretty large training
% data set because it has many parameters to learn. Instead, Matlab's
% Econometrics Toolbox has a framework for learning AR models.
Mdl = arima(3,0,0); % <-- this 3 means we are training a 3rd order model
Mdl = estimate(Mdl,sequence);
% The model coefficients may not be perfect, but they are the best fit to
% the noisy training data provided.

% This trained model can be used to predict the next symbol in a sequence.
% For example, suppose sym(n-1) = 5, sym(n-2) = 3, and sym(n-3) = 1. Then 
% we can predict sym(n) as follows:
[predictedSym,~,V] = forecast(Mdl,1,[1;3;5]);
predictedSym

% We need to transform this predicted symbol 
% (which is not even an integer) into a probabilistic forecast for the 
% next symbol. The AR model assumes a Gaussian distribution centered 
% at predictedSym with variance V.
% 
% I recommend using the command normpdf to get 9 samples of that Gaussian
% pdf, and then normalize those 9 numbers so they sum to 1. Note: normpdf 
% requires the standard deviation, not the variance!

load sequence_ElecDemand_train.mat % allowed because it is TRAINING data
%sequenceLength = length(sequence);
Mdl = arima(2,0,0);
Mdl = estimate(Mdl,sequence);
sequenceLength = initializeSymbolMachineS26('sequence_ElecDemand_test.mat');
probs = [1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9];
[past2,~] = symbolMachineS26(probs);
[past1,~] = symbolMachineS26(probs);
for ii = 3:sequenceLength
 [predictedSym,~,V] = forecast(Mdl,1,[past2;past1]); 
 probs = normpdf(1:9,predictedSym,sqrt(V));
 probs = probs/sum(probs);
 [symbol,~] = symbolMachineS26(probs);
 past2 = past1;
 past1 = symbol;
end
reportSymbolMachineS26;
% 2) Compare the performance of your second-order AR model to a
% second-order MC model (trained on the ElecDemand training dataset, 
% and tested on the ElecDemand testing dataset).
%sequenceLength = length(sequence);
sequenceLength = initializeSymbolMachineS26('sequence_ElecDemand_test.mat');
symbolCounts = ones(9,9,9); % Prevents any 0 probability forecasts
for ii = 3:sequenceLength
 currentSymbol = sequence(ii);
 past1 = sequence(ii-1);
 past2 = sequence(ii-2);
 symbolCounts(past2,past1,currentSymbol) = ...
 symbolCounts(past2,past1,currentSymbol) + 1;
end
probMatrix = symbolCounts;
for past2 = 1:9
 for past1 = 1:9
 probMatrix(past2,past1,:) = probMatrix(past2,past1,:)/sum(probMatrix(past2,past1,:));
 end
end
probs = [1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9];
[past2,~] = symbolMachineS26(probs);
[past1,~] = symbolMachineS26(probs);
for ii = 3:sequenceLength
 [symbol,~] = symbolMachineS26(probMatrix(past2, past1,:));
 past2 = past1;
 past1 = symbol;
end
reportSymbolMachineS26;
% 3) Train a FOURTH-order AR model on the ElecDemand training dataset, 
% and use that model to forecast the ElecDemand testing dataset.
Mdl = arima(4,0,0);
Mdl = estimate(Mdl,sequence);
sequenceLength = initializeSymbolMachineS26('sequence_ElecDemand_test.mat');
probs = [1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9];
[past4,~] = symbolMachineS26(probs);
[past3,~] = symbolMachineS26(probs);
[past2,~] = symbolMachineS26(probs);
[past1,~] = symbolMachineS26(probs);
for ii = 5:sequenceLength
 [predictedSym,~,V] = forecast(Mdl,1,[past4;past3;past2;past1]); 
 % Compute Gaussian pdf values at integers 1:9 using predicted mean and std
 probs = normpdf(1:9,predictedSym,sqrt(V));
 probs = probs/sum(probs);
 [symbol,~] = symbolMachineS26(probs);
 past4 = past3;
 past3 = past2;
 past2 = past1;
 past1 = symbol;
end
reportSymbolMachineS26;