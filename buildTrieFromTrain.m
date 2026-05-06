function [trieChildren, trieCounts, nodeCount, priorProb, trainSeq] = buildTrieFromTrain(trainFile, k)
%BUILDTRIEFROMTRAIN Construct a suffix trie from a training .mat sequence file.
%   [trieChildren, trieCounts, nodeCount, priorProb, trainSeq] =
%   buildTrieFromTrain(trainFile, k)

if nargin < 2
    error('buildTrieFromTrain requires trainFile and k');
end

s = load(trainFile);
if isfield(s, 'sequence')
    trainSeq = s.sequence(:)';
else
    error('buildTrieFromTrain:MissingSequence', 'The file %s does not contain ''sequence''.', trainFile);
end

N = length(trainSeq);

% Preallocate using same heuristic as expmodelTrie.m
trieMaxFromN    = 1 + max(0, N * k) * k; % conservative reservation for standalone builder
trieMaxFromTree = floor((9^(k + 1) - 1) / 8);
maxNodes        = min(trieMaxFromN, trieMaxFromTree) + 2;

trieChildren = zeros(maxNodes, 9, 'uint32');
trieCounts   = zeros(maxNodes, 9);
nodeCount    = uint32(1); % root = 1

% Build trie: for each position i in training sequence, extract preceding k symbols
for i = k + 1:N
    next_sym = trainSeq(i);
    contextAtI = trainSeq(i-k:i-1);
    node = uint32(1);
    for d = 1:k
        sym = contextAtI(end - d + 1);
        if trieChildren(node, sym) == 0
            nodeCount = nodeCount + 1;
            trieChildren(node, sym) = nodeCount;
        end
        node = trieChildren(node, sym);
        trieCounts(node, next_sym) = trieCounts(node, next_sym) + 1;
    end
end

% Global prior
priorCounts = zeros(1,9);
for sym = 1:9
    priorCounts(sym) = sum(trainSeq == sym);
end
priorProb = priorCounts / sum(priorCounts);
end
