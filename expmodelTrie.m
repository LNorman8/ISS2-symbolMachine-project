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
k = 8; % Max context depth, matching the baseline expWeightmodel
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
trieMaxFromN    = 1 + max(0, N - k) * k;
trieMaxFromTree = floor((9^(k + 1) - 1) / 8);
maxNodes        = min(trieMaxFromN, trieMaxFromTree) + 2;
trieChildren = zeros(maxNodes, 9, 'uint32');
trieCounts   = zeros(maxNodes, 9);
nodeCount    = uint32(1); % root = node 1
for i = k + 1:N
    next_sym = trainSeq(i);
    node = uint32(1); % start at root
    for d = 1:k
        sym = trainSeq(i - d); % d-th most recent symbol before position i
        if trieChildren(node, sym) == 0
            nodeCount = nodeCount + 1;
            trieChildren(node, sym) = nodeCount;
        end
        node = trieChildren(node, sym);
        trieCounts(node, next_sym) = trieCounts(node, next_sym) + 1;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- TEST SET SETUP (Symbol Machine) ---
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T = initializeSymbolMachineS26(testFile, 0);
% Bootstrap the initial context from the end of the training sequence so
% that the first test predictions already have a full k-symbol context.
context = trainSeq(end-k+1:end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- VARIABLE-LENGTH FORECAST LOOP ---
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exponential depth weighting: each depth d contributes 3^d times its raw
% trie counts to a shared accumulator.  Deeper (more specific) contexts
% therefore dominate the final distribution, mirroring the weight = 3^L
% scheme in the baseline expWeightmodel but using an O(k) trie traversal
% instead of an O(N*k) training-set scan.
for t = 1:T
    % Unigram baseline (same scaling as expWeightmodel's priorProb * 0.5)
    accumCounts = priorProb * 0.5;
    node = uint32(1); % root
    for d = 1:k
        sym = context(end - d + 1); % d-th most recent symbol
        childNode = trieChildren(node, sym);
        if childNode == 0
            break; % context unseen at this depth; stop early
        end
        node = childNode;
        % Exponentially higher reward for deeper context matches
        accumCounts = accumCounts + (3^d) * trieCounts(node, :);
    end
    % ============================================================
    % Normalise to a valid probability vector
    % ============================================================
    probVec = accumCounts / sum(accumCounts);
    % ============================================================
    % Symbol Machine step (penalty is accumulated internally in SYMBOLDATA)
    % ============================================================
    [symbol, ~] = symbolMachineS26(probVec);
    % ============================================================
    % Update context (sliding window)
    % ============================================================
    context = [context(2:end), symbol];
end
reportSymbolMachineS26;
end