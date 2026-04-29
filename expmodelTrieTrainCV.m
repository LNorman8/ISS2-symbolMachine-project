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

function bitsCV = expmodelTrieTrainCV(name, k, weightBase, priorScale, gamma, foldFrac)

if nargin < 6
    foldFrac = 0.2;  % Use 20% for validation, 80% for training
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

context = zeros(1, k);
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

    % Update context
    if i >= k
        context = trainSeq(i-k+1:i);
    else
        context = [zeros(1, k-i), trainSeq(1:i)];
    end
end

% ============================================================
% 2. Evaluate on validation portion
% ============================================================
totalBits = 0;
valLen = length(valSeq);
context = zeros(1, k);

for i = 1:valLen
    sym = valSeq(i);

    % Forecast using trie
    accumProb = priorProb * priorScale;
    node = uint32(1);

    for d = 1:k
        if d <= i + nTrain  % Account for position in full sequence
            if d <= i
                ctxSym = valSeq(i - d + 1);
            else
                ctxSym = trainSeq(end - (d - i) + 1);
            end
        else
            ctxSym = 0;
        end

        if ctxSym > 0 && ctxSym <= 9
            childNode = trieChildren(node, ctxSym);
            if childNode == 0
                break;
            end
            node = childNode;
            depthCounts = trieCounts(node, :);
            depthTotal = sum(depthCounts);

            if depthTotal > 0
                lambda = depthTotal / (depthTotal + gamma);
                depthProb = lambda * (depthCounts / depthTotal) + (1 - lambda) * priorProb;
            else
                depthProb = priorProb;
            end

            accumProb = accumProb + (weightBase^d) * depthProb;
        else
            break;
        end
    end

    % Normalize and compute loss
    accumProb = accumProb / sum(accumProb);
    prob = max(accumProb(sym), realmin);
    totalBits = totalBits - log2(prob);

    % Update context for next symbol
    if i >= k
        context = valSeq(i-k+1:i);
    else
        context = [zeros(1, k-i), valSeq(1:i)];
    end
end

bitsCV = totalBits / valLen;

end

function [trieChildren, trieCounts] = growTrieStorage(trieChildren, trieCounts, requiredNodeCount)
currentSize = size(trieChildren, 1);
targetSize = max(currentSize * 2, double(requiredNodeCount) + 1024);
trieChildren(targetSize, 9) = uint32(0);
trieCounts(targetSize, 9) = 0;
end
