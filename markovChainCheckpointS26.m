%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MARKOV CHAIN PROJECT CHECKPOINT
%%% Demonstration of Markov Chain models up to second order.
%%% Colorado School of Mines EENG311 - Spring 2026 - Mike Wakin
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Let's use some real data: text from an 1843 algebra book printed 
% in the Hawaiian language. To make this compatible with the Symbol
% Machine, the text has been converted into valid symbols (1-9):
% 
%  1 = vowels (A,E,I,O,U)
%  2 = H
%  3 = K
%  4 = L
%  5 = M
%  6 = N
%  7 = P
%  8 = W
%  9 = all other characters including spaces. 
%
% The training sequence contains 22860 characters; recall that we are
% free to load that directly and train any model we want. The testing 
% sequence then contains 5654 characters.

clear all;
load sequence_Hawaiian_train.mat; % OK because this is TRAINING data
sequenceLength = length(sequence);
estimatedPMF = zeros(1,9);
for symbol = 1:9
    estimatedPMF(symbol) = sum(sequence==symbol);
end
estimatedPMF = estimatedPMF/sequenceLength
sequenceLength = initializeSymbolMachineS26('sequence_Hawaiian_train.mat',0);
for ii = 1:sequenceLength
    [symbol,penalty] = symbolMachineS26(estimatedPMF);
end
reportSymbolMachineS26;

% [-- First-order Markov Chain --]
% A first-order MC uses the one most recent symbol to make a conditional
% probability forecast for the next symbol. To train such a model,
% just like we did with the Dickens corpus, we can learn a conditional
% probability model (from the training data) for the pmf of a symbol
% conditioned on the symbol that preceded it.
clear all;
load sequence_Hawaiian_train.mat; % OK because this is TRAINING data
sequenceLength = length(sequence);
symbolCounts = ones(9,9); % Prevents any 0 probability forecasts
for ii = 2:sequenceLength
    currentSymbol = sequence(ii);
    precedingSymbol = sequence(ii-1);
    symbolCounts(precedingSymbol,currentSymbol) = ...
        symbolCounts(precedingSymbol,currentSymbol) + 1;
end
probMatrix = symbolCounts;
for ii = 1:9
    probMatrix(ii,:) = probMatrix(ii,:)/sum(probMatrix(ii,:));
end
% Each row of probMatrix is a conditional pmf (which sums to 1)
figure;clf;
imagesc(probMatrix);colorbar;title('Learned from training data');
ylabel('preceding symbol');
xlabel('forecasted symbol');
% Let's use the first-order MC model for forecasting the training data.
% On the Hawaiian training data, this outperforms the 0th-order MC model.
sequenceLength = initializeSymbolMachineS26('sequence_Hawaiian_train.mat',0);
% Start with a uniform forecast for the first symbol
probs = [1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9];
[symbol,penalty] = symbolMachineS26(probs);
for ii = 2:sequenceLength
    % For each subsequent symbol, we can base our forecast on the 
    % preceding symbol (which was given to us by the Symbol Machine)
    [symbol,penalty] = symbolMachineS26(probMatrix(symbol,:));
end
reportSymbolMachineS26;

% [-- Second-order Markov Chain --]
% A second-order MC uses the two most recent symbols to make a conditional
% probability forecast for the next symbol. To train such a model, we 
% must learn a conditional probability model (from the training data) 
% for the pmf of a symbol conditioned on the two symbols that preceded it.
clear all;
load sequence_Hawaiian_train.mat; % OK because this is TRAINING data
sequenceLength = length(sequence);
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
% This second-order MC model should outperform the first-order MC 
% model on the Hawaiian training dataset. 
sequenceLength = initializeSymbolMachineS26('sequence_Hawaiian_train.mat',0);
probs = [1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9];
[past2,penalty] = symbolMachineS26(probs);
[past1,penalty] = symbolMachineS26(probs);
for ii = 3:sequenceLength
    [symbol,~] = symbolMachineS26(probMatrix(past2, past1,:));
    past2 = past1;
    past1 = symbol;
end
reportSymbolMachineS26;