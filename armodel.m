% third order autoregressive model
function [] = armodel(name)
arguments
    name (1,:) char = "ElecDemand"
end
trainFile = ['sequence_' name '_train.mat'];
testFile  = ['sequence_' name '_test.mat'];
seq_struct = load(trainFile);
sequence = seq_struct.sequence;
Mdl = arima(4,0,0);
Mdl = estimate(Mdl,sequence);
sequenceLength = initializeSymbolMachineS26(testFile, 0);
probs = [1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9 1/9];
[past4,~] = symbolMachineS26(probs);
[past3,~] = symbolMachineS26(probs);
[past2,~] = symbolMachineS26(probs);
[past1,~] = symbolMachineS26(probs);
for ii = 5:sequenceLength
 [predictedSym,~,V] = forecast(Mdl,1,[past4;past3;past2;past1]);
 probs = normpdf(1:9,predictedSym,sqrt(V));
 probs = probs/sum(probs);
 [symbol,~] = symbolMachineS26(probs);
 past4 = past3;
 past3 = past2;
 past2 = past1;
 past1 = symbol;
end
reportSymbolMachineS26;
end