%%% TRAINING-ONLY HYPERPARAMETER EVALUATION FOR EXPMODELTRIE
% Evaluates expmodelTrie hyperparameters using cross-validation on training
% data ONLY, without touching test data during tuning.
%
% Usage: bitsCV = expmodelTrieTrainCV(name, k, weightBase, priorScale, gamma, foldFrac)
%   name: dataset name (e.g., 'Hawaiian')
%   k, weightBase, priorScale, gamma: hyperparameters to evaluate
%   foldFrac: fraction of training data to use for validation (default: 0.2)
%
% Returns: average bits per symbol on validation fold (cross-validation score)

function bitsCV = expmodelTrieTrainCV(name, k, weightBase, priorScale, gamma, foldFrac, mixAlpha, shallowK)

if nargin < 6
    foldFrac = 0.2;  % Use 20% for validation, 80% for training
end
if nargin < 7 || isempty(mixAlpha)
    mixAlpha = defaultMixAlpha(k);
end
if nargin < 8 || isempty(shallowK)
    shallowK = defaultShallowK(k);
end

trainFile = ['sequence_' name '_train.mat'];
seq_struct = load(trainFile);
sequence = seq_struct.sequence(:)';

% Split training data into train and validation
N = length(sequence);
nVal = max(1, round(N * foldFrac));
nTrain = N - nVal;

trainSeq = sequence(1:nTrain);
valSeq = sequence(nTrain+1:end);

% ============================================================
% 1. Build trie on training portion only
% ============================================================
priorCounts = zeros(1, 9);
for sym = 1:9
    priorCounts(sym) = sum(trainSeq == sym);
end
priorProb = priorCounts / sum(priorCounts);

% Build trie from training portion
% Start with a moderate allocation and grow as needed to avoid hard caps.
maxNodes = min(max(nTrain, 50000), 500000);
trieChildren = zeros(maxNodes, 9, 'uint32');
trieCounts = zeros(maxNodes, 9);
nodeCount = uint32(1);

for i = 1:length(trainSeq)
    sym = trainSeq(i);

    % Add to trie
    node = uint32(1);
    for d = 1:k
        if d <= i
            ctxSym = trainSeq(i - d + 1);
        else
            ctxSym = 0;
        end

        if ctxSym > 0
            if trieChildren(node, ctxSym) == 0
                nodeCount = nodeCount + 1;
                if double(nodeCount) > size(trieChildren, 1)
                    [trieChildren, trieCounts] = growTrieStorage(trieChildren, trieCounts, nodeCount);
                end
                trieChildren(node, ctxSym) = nodeCount;
            end
            node = trieChildren(node, ctxSym);
            trieCounts(node, sym) = trieCounts(node, sym) + 1;
        end
    end

end

% ============================================================
% 2. Evaluate on validation portion
% ============================================================
totalBits = 0;
valLen = length(valSeq);
fullSeq = [trainSeq valSeq];

for i = 1:valLen
    sym = valSeq(i);

    % Forecast using trie
    histEnd = nTrain + i - 1;
    histStart = max(1, histEnd - k + 1);
    context = fullSeq(histStart:histEnd);
    if numel(context) < k
        context = [zeros(1, k - numel(context)), context];
    end

    probVec = forecastFromTrieMixture(context, trieChildren, trieCounts, ...
        priorProb, k, weightBase, priorScale, gamma, mixAlpha, shallowK);
    prob = max(probVec(sym), realmin);
    totalBits = totalBits - log2(prob);

    % Update context for next symbol by learning from the observed validation symbol.
    priorCounts(sym) = priorCounts(sym) + 1;
    priorProb = priorCounts / sum(priorCounts);
    [trieChildren, trieCounts, nodeCount] = updateTrieWithObservation( ...
        trieChildren, trieCounts, nodeCount, context, sym, k);
end

bitsCV = totalBits / valLen;

end

function probVec = forecastFromTrieMixture(context, trieChildren, trieCounts, ...
    priorProb, k, weightBase, priorScale, gamma, mixAlpha, shallowK)
deepProb = forecastFromTrie(context, trieChildren, trieCounts, priorProb, k, weightBase, priorScale, gamma);
if isempty(mixAlpha) || mixAlpha <= 0 || isempty(shallowK) || shallowK >= k
    probVec = deepProb;
    return;
end

shallowProb = forecastFromTrie(context, trieChildren, trieCounts, priorProb, shallowK, weightBase, priorScale, gamma);
probVec = mixAlpha * shallowProb + (1 - mixAlpha) * deepProb;
probVec = probVec / sum(probVec);
probVec = probVec(:)';
end

function probVec = forecastFromTrie(context, trieChildren, trieCounts, ...
    priorProb, k, weightBase, priorScale, gamma)
accumProb = priorProb * priorScale;
node = uint32(1);
for d = 1:k
    sym = context(end - d + 1);
    childNode = trieChildren(node, sym);
    if childNode == 0
        break;
    end

    node = childNode;
    depthCounts = trieCounts(node, :);
    total = sum(depthCounts);
    if total > 0
        lambda = total / (total + gamma);
        depthProb = lambda * (depthCounts / total) + (1 - lambda) * priorProb;
    else
        depthProb = priorProb;
    end

    accumProb = accumProb + (weightBase^d) * depthProb;
end

probVec = accumProb / sum(accumProb);
probVec = probVec(:)';
end

function [trieChildren, trieCounts] = growTrieStorage(trieChildren, trieCounts, requiredNodeCount)
currentSize = size(trieChildren, 1);
targetSize = max(currentSize * 2, double(requiredNodeCount) + 1024);
trieChildren(targetSize, 9) = uint32(0);
trieCounts(targetSize, 9) = 0;
end

function mixAlpha = defaultMixAlpha(k)
if k >= 4
    mixAlpha = 0.25;
else
    mixAlpha = 0.0;
end
end

function shallowK = defaultShallowK(k)
shallowK = min(2, max(1, k - 1));
end

function [trieChildren, trieCounts, nodeCount] = updateTrieWithObservation( ...
    trieChildren, trieCounts, nodeCount, context, nextSym, k)

node = uint32(1);
for d = 1:k
    sym = context(end - d + 1);
    if trieChildren(node, sym) == 0
        nodeCount = nodeCount + 1;
        if double(nodeCount) > size(trieChildren, 1)
            [trieChildren, trieCounts] = growTrieStorage(trieChildren, trieCounts, nodeCount);
        end
        trieChildren(node, sym) = nodeCount;
    end
    node = trieChildren(node, sym);
    trieCounts(node, nextSym) = trieCounts(node, nextSym) + 1;
end
end
