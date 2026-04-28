%% 4th order markov chain model

function [] = markovmodel(name)
arguments
    name (1,:) char = "Hawaiian"
end
trainFile = ['sequence_' name '_train.mat'];
testFile  = ['sequence_' name '_test.mat'];
seq_struct = load(trainFile);
sequence = seq_struct.sequence; % Extract the sequence from the loaded data
sequenceLength = length(sequence);
symbolCounts = ones(9,9,9,9,9); % Prevents any 0 probability forecasts
for ii = 5:sequenceLength
    currentSymbol = sequence(ii);
    past1 = sequence(ii-1);
    past2 = sequence(ii-2);
    past3 = sequence(ii-3);
    past4 = sequence(ii-4);
    symbolCounts(past4,past3,past2,past1,currentSymbol) = ...
        symbolCounts(past4,past3,past2,past1,currentSymbol) + 1;
end
probMatrix = symbolCounts;
for past4 = 1:9
    for past3 = 1:9
        for past2 = 1:9
            for past1 = 1:9
                probMatrix(past4,past3,past2,past1,:) = probMatrix(past4,past3,past2,past1,:)/sum(probMatrix(past4,past3,past2,past1,:));
            end
        end
    end
end
sequenceLength = initializeSymbolMachineS26(testFile, 0);
probs = [1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9];
[past4,~] = symbolMachineS26(probs);
[past3,~] = symbolMachineS26(probs);
[past2,~] = symbolMachineS26(probs);
[past1,~] = symbolMachineS26(probs);
for ii = 5:sequenceLength
    [symbol,~] = symbolMachineS26(probMatrix(past4, past3, past2, past1,:));
    past4 = past3;
    past3 = past2;
    past2 = past1;
    past1 = symbol;
end
reportSymbolMachineS26;
end