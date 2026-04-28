%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VARIABLE-CONTEXT MODEL WITH TRIE LOOKUP AND EXPONENTIAL DEPTH WEIGHTING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = expmodelTrie(name)
arguments
    name (1,:) char = 'Hawaiian'
end
trainFile = ['sequence_' name '_train.mat'];
testFile  = ['sequence_' name '_test.mat'];
seq_struct = load(trainFile);
sequence = seq_struct.sequence; % Extract the sequence from the loaded data
trainSeq = sequence(:)'; % Ensure row vector
N = length(trainSeq);
% Test sequence length is needed up front so the trie can reserve room for
% online updates while forecasting.
T = initializeSymbolMachineS26(testFile, 0);

% Candidate grids for depth weighting and smoothing.
maxK = min(8, max(1, N - 1));
weightBaseCands = [1.8, 2.2, 2.6, 3.0];
priorScaleCands = [0.2, 0.5, 1.0];
gammaCands = [4.0, 8.0, 16.0];

% Select context depth and smoothing parameters with an inner validation
% pass on training data.
[k, weightBase, priorScale, gamma] = chooseAdaptiveParams(trainSeq, ...
    maxK, weightBaseCands, priorScaleCands, gammaCands);
% ============================================================
% 1. Precompute global prior (marginal distribution)
% ============================================================
priorCounts = zeros(1, 9);
for sym = 1:9
    priorCounts(sym) = sum(trainSeq == sym);
end
priorProb = priorCounts / sum(priorCounts);
% ============================================================
% 2. Build suffix trie over training sequence
% ============================================================
% The trie is keyed by reversed context (most-recent symbol first).
%   trieChildren(node, sym) = child node index (uint32; 0 = no child)
%   trieCounts(node, sym)   = count of next symbol sym after this context
%
% Root node (index 1) represents the 0-length context; the unigram
% priorProb is used directly as the depth-0 base (no counts stored there).
%
% A node reached via symbols s1, s2, ..., sd from root represents the
% context "most-recent = s1, second-most-recent = s2, ..., d-th = sd".
trieMaxFromN    = 1 + max(0, N + T - k) * k;
trieMaxFromTree = floor((9^(k + 1) - 1) / 8);
maxNodes        = min(trieMaxFromN, trieMaxFromTree) + 2;
trieChildren = zeros(maxNodes, 9, 'uint32');
trieCounts   = zeros(maxNodes, 9);
nodeCount    = uint32(1); % root = node 1
for i = k + 1:N
    next_sym = trainSeq(i);
    contextAtI = trainSeq(i-k:i-1);
    [trieChildren, trieCounts, nodeCount] = updateTrieWithObservation( ...
        trieChildren, trieCounts, nodeCount, contextAtI, next_sym, k);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- TEST SET SETUP (Symbol Machine) ---
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bootstrap the initial context from the end of the training sequence so
% that the first test predictions already have a full k-symbol context.
context = trainSeq(end-k+1:end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- VARIABLE-LENGTH FORECAST LOOP ---
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exponential depth weighting with node-level JM smoothing. During testing,
% each observed symbol is fed back into the trie so the model keeps
% learning online.
for t = 1:T
    probVec = forecastFromTrie(context, trieChildren, trieCounts, ...
        priorProb, k, weightBase, priorScale, gamma);
    % ============================================================
    % Symbol Machine step (penalty is accumulated internally in SYMBOLDATA)
    % ============================================================
    [symbol, ~] = symbolMachineS26(probVec);

    % Online adaptation: absorb the observed symbol into prior and trie.
    priorCounts(symbol) = priorCounts(symbol) + 1;
    priorProb = priorCounts / sum(priorCounts);
    [trieChildren, trieCounts, nodeCount] = updateTrieWithObservation( ...
        trieChildren, trieCounts, nodeCount, context, symbol, k);

    % ============================================================
    % Update context (sliding window)
    % ============================================================
    context = [context(2:end), symbol];
end
reportSymbolMachineS26;
end

function [kBest, weightBaseBest, priorScaleBest, gammaBest] = ...
    chooseAdaptiveParams(trainSeq, maxK, weightBaseCands, priorScaleCands, gammaCands)
N = length(trainSeq);
if N <= 40
    kBest = min(4, maxK);
    weightBaseBest = weightBaseCands(min(2, numel(weightBaseCands)));
    priorScaleBest = priorScaleCands(min(2, numel(priorScaleCands)));
    gammaBest = gammaCands(min(2, numel(gammaCands)));
    return;
end

% Use a representative tail window for adaptive-k search so very large
% datasets remain practical to evaluate repeatedly during tuning.
cvTotalCap = 12000;
if N > cvTotalCap
    seqForCV = trainSeq(end-cvTotalCap+1:end);
else
    seqForCV = trainSeq;
end

splitIdx = floor(0.7 * length(seqForCV));
splitIdx = max(splitIdx, 20);
splitIdx = min(splitIdx, length(seqForCV) - 10);
trainPart = seqForCV(1:splitIdx);
valPart = seqForCV(splitIdx + 1:end);

kBest = 1;
bestBits = Inf;
weightBaseBest = weightBaseCands(1);
priorScaleBest = priorScaleCands(1);
gammaBest = gammaCands(1);
for kCand = 1:maxK
    if length(trainPart) <= kCand || isempty(valPart)
        continue;
    end

    for wIdx = 1:numel(weightBaseCands)
        for pIdx = 1:numel(priorScaleCands)
            for gIdx = 1:numel(gammaCands)
                weightBase = weightBaseCands(wIdx);
                priorScale = priorScaleCands(pIdx);
                gamma = gammaCands(gIdx);

                priorCounts = zeros(1, 9);
                for sym = 1:9
                    priorCounts(sym) = sum(trainPart == sym);
                end
                priorProb = priorCounts / sum(priorCounts);

                maxNodesCV = 1 + max(0, length(trainPart) + length(valPart) - kCand) * kCand + 2;
                trieChildren = zeros(maxNodesCV, 9, 'uint32');
                trieCounts = zeros(maxNodesCV, 9);
                nodeCount = uint32(1);
                for i = kCand + 1:length(trainPart)
                    nextSym = trainPart(i);
                    contextAtI = trainPart(i-kCand:i-1);
                    [trieChildren, trieCounts, nodeCount] = updateTrieWithObservation( ...
                        trieChildren, trieCounts, nodeCount, contextAtI, nextSym, kCand);
                end

                context = trainPart(end-kCand+1:end);
                lossBits = 0;
                for i = 1:length(valPart)
                    trueSym = valPart(i);
                    probVec = forecastFromTrie(context, trieChildren, trieCounts, ...
                        priorProb, kCand, weightBase, priorScale, gamma);
                    lossBits = lossBits - log2(max(probVec(trueSym), realmin));

                    priorCounts(trueSym) = priorCounts(trueSym) + 1;
                    priorProb = priorCounts / sum(priorCounts);
                    [trieChildren, trieCounts, nodeCount] = updateTrieWithObservation( ...
                        trieChildren, trieCounts, nodeCount, context, trueSym, kCand);
                    context = [context(2:end), trueSym];
                end

                bitsPerSymbol = lossBits / length(valPart);
                if bitsPerSymbol < bestBits
                    bestBits = bitsPerSymbol;
                    kBest = kCand;
                    weightBaseBest = weightBase;
                    priorScaleBest = priorScale;
                    gammaBest = gamma;
                end
            end
        end
    end
end
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

function [trieChildren, trieCounts, nodeCount] = updateTrieWithObservation( ...
    trieChildren, trieCounts, nodeCount, context, nextSym, k)

node = uint32(1);
for d = 1:k
    sym = context(end - d + 1);
    if trieChildren(node, sym) == 0
        nodeCount = nodeCount + 1;
        trieChildren(node, sym) = nodeCount;
    end
    node = trieChildren(node, sym);
    trieCounts(node, nextSym) = trieCounts(node, nextSym) + 1;
end
end