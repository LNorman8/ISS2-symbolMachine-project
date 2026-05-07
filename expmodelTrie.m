%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VARIABLE-CONTEXT MODEL WITH TRIE LOOKUP AND EXPONENTIAL DEPTH WEIGHTING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = expmodelTrie(name, k, weightBase, priorScale, gamma)
arguments
    name (1,:) char = 'Hawaiian'
    k (1,1) double = NaN  % Use NaN to indicate "not provided"; provide value to skip tuning
    weightBase (1,1) double = NaN  % Must provide all four or none
    priorScale (1,1) double = NaN
    gamma (1,1) double = NaN
end

% VALIDATION: Check that either all four parameters are provided or none are.
% Users cannot partially specify parameters; this prevents accidental misuse.
providedCount = sum([~isnan(k), ~isnan(weightBase), ~isnan(priorScale), ~isnan(gamma)]);
if providedCount > 0 && providedCount < 4
    error('expmodelTrie:PartialParams', ...
        'All four parameters (k, weightBase, priorScale, gamma) must be provided together, or none to enable AUTOTUNE.');
end
userProvidedParams = (providedCount == 4);

% ALGORITHM OVERVIEW:
%   Builds a suffix trie over training sequence with exponential depth weighting
%   to predict next symbol. Each node represents a context; transitions are weighted
%   by (weightBase^depth) so deeper contexts have more influence. Uses Jelinek-Mercer
%   smoothing (parameter gamma) to blend empirical counts with global prior at each
%   node. During test set forecasting, observed symbols are fed back online into
%   both the trie and prior counts, allowing the model to adapt to test distribution.
%
%   USAGE:
%     expmodelTrie('Hawaiian')  % Enable automatic parameter tuning
%     expmodelTrie('Hawaiian', 8, 1.25, 0.75, 0.0)  % Use provided: k, weightBase, priorScale, gamma
%
trainFile = ['./data/sequence_' name '_train.mat'];
testFile  = ['./data/sequence_' name '_test.mat'];
seq_struct = load(trainFile);
sequence = seq_struct.sequence; % Extract the sequence from the loaded data
trainSeq = sequence(:)'; % Ensure row vector
N = length(trainSeq);

% PARAMETER TUNING:
%   Uses inner cross-validation loop (70% train / 30% validation split on training data)
%   to search for best (k, weightBase, priorScale, gamma). Evaluates each candidate
%   by building a trie on train part, then computing bits-per-symbol on val part.
%   Dataset-specific candidate grids (ULTRATUNE=true) vs. wide general defaults.
%
%   - k: Max context depth. Trade-off: deeper = more context but more overfitting risk.
%   - weightBase: Exponential base for depth weighting. >1 emphasizes deeper contexts;
%       =1 means all depths equally weighted. <1 emphasizes shallower contexts. 
%       <1 feels like admitting defeat for our approach, but if it works it works.
%   - priorScale: Scales the global prior contribution. High values make model rely
%       more on overall symbol frequency (useful when data is sparse).
%   - gamma: Jelinek-Mercer smoothing. Higher values interpolate empirical counts
%       toward the prior, reducing overfitting on low-count nodes.
%
if userProvidedParams
    % Use user-provided values exactly as entered.
    paramSource = 'USER-PROVIDED';
else
    switch name
        case 'Hawaiian' % 1.3128
            k = 28;
            weightBase = 1.015;
            priorScale = 0.240;
            gamma = 0.00;
        case 'Dickens'
            k = 9;
            weightBase = 2.00;
            priorScale = 0.10;
            gamma = 2.00;
        case 'DIAtemp'
            k = 4;
            weightBase = 0.75; % Tuning decided that the wind is unpredictable
            priorScale = 0.01;
            gamma = 0.90;
        case 'DIAwind'
            k = 10;
            weightBase = 0.28;
            priorScale = 0.045;
            gamma = 7.375;
        case 'ElecDemand'
            k = 12;
            weightBase = 0.96;
            priorScale = 0.06;
            gamma = 0.00;
        case 'HoustonRain'
            k = 10;
            weightBase = 1.25;
            priorScale = 1.50;
            gamma = 12.00;
        case 'solarWind'
            k = 7;
            weightBase = 0.86;
            priorScale = 0.0065;
            gamma = 0.0001;
    end
    paramSource = 'PRESET';
end

% Display chosen params and their source
fprintf('Chosen params for %s [%s]: k=%d, weightBase=%.3f, priorScale=%.3f, gamma=%.3f\n', ...
    name, paramSource, k, weightBase, priorScale, gamma);
% ============================================================
% 1. Precompute global prior (marginal distribution)
% ============================================================
priorCounts = zeros(1, 9);
for sym = 1:9
    priorCounts(sym) = sum(trainSeq == sym);
end
priorProb = priorCounts / sum(priorCounts);

T = initializeSymbolMachineS26(testFile, 0);
% ============================================================
% 2. Build suffix trie over training sequence
% ============================================================
% TRIE DATA STRUCTURE:
%   - trieChildren(node, sym) = child node index (0 = no child)
%   - trieCounts(node, sym)   = count of next symbol sym observed after context
%
%   Root node 1 = empty context (depth 0); priorProb is used directly, no counts.
%
%   KEY INSIGHT: Context is stored in REVERSE order (most-recent symbol first).
%   A node reached by traversing s1→s2→...→sd from root represents the context
%   [recent, ..., older] = [s1, s2, ..., sd], where s1 was just observed.
trieMaxFromN    = 1 + max(0, N + T - k) * k;
trieMaxFromTree = floor((9^(k + 1) - 1) / 8);
maxNodes        = min(trieMaxFromN, trieMaxFromTree) + 2;
trieChildren = zeros(maxNodes, 9, 'uint32');
trieCounts   = zeros(maxNodes, 9);
nodeCount    = uint32(1); % root = node 1

% TRIE BUILDING:
%   For each position i in training sequence, extract the preceding k symbols
%   as context and observe the next symbol (trainSeq(i)). Navigate the trie
%   along this context path, creating nodes as needed, and increment the count
%   for the observed next symbol at the final node.
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
for t = 1:T
    probVec = forecastFromTrie(context, trieChildren, trieCounts, ...
        priorProb, k, weightBase, priorScale, gamma);
    % Step Symbol Machine (penalty is accumulated internally in SYMBOLDATA). Still feel like there's gotta be a use for the penalty return
    [symbol, ~] = symbolMachineS26(probVec);

    % Online adaptation: absorb the observed symbol into prior and trie.
    priorCounts(symbol) = priorCounts(symbol) + 1;
    priorProb = priorCounts / sum(priorCounts);
    [trieChildren, trieCounts, nodeCount] = updateTrieWithObservation( ...
        trieChildren, trieCounts, nodeCount, context, symbol, k);

    % Slide Window
    context = [context(2:end), symbol];
end
reportSymbolMachineS26;
end

function probVec = forecastFromTrie(context, trieChildren, trieCounts, ...
    priorProb, k, weightBase, priorScale, gamma)
%
% PREDICT NEXT SYMBOL BY WALKING TRIE AND ACCUMULATING WEIGHTED PROBABILITIES:
%   - Start with priorProb scaled by priorScale (depth-0 base).
%   - For each depth d = 1 to k, navigate trie using context(end - d + 1)
%       (extracts symbols in reverse order to match trie's reverse-context indexing).
%   - At each node, extract symbol counts and apply Jelinek-Mercer smoothing:
%      - lambda = total_count / (total_count + gamma)
%      - depthProb = lambda * empirical + (1 - lambda) * priorProb
%      - Accumulate weighted probability: (weightBase^d) * depthProb
%      - Higher weightBase emphasizes deeper (more specific) contexts.
%      - Defensive checks: clamp negative/NaN/Inf values, handle zero-sum cases.
%      - Normalize final accumulated probability vector to sum to 1.
%
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

% Defensive checks: prevent zero-sum, NaN or Inf which would produce NaNs
accumProb(~isfinite(accumProb) | accumProb < 0) = 0;
s = sum(accumProb);
if s <= 0 || ~isfinite(s)
    % Fall back to a tiny-smoothed global prior to avoid poisoning counts
    accumProb = priorProb + realmin;
    s = sum(accumProb);
end

probVec = accumProb / s;
probVec = probVec(:)';
end

function [trieChildren, trieCounts, nodeCount] = updateTrieWithObservation( ...
    trieChildren, trieCounts, nodeCount, context, nextSym, k)
%
% UPDATE TRIE WITH A NEW OBSERVATION:
%   Given a context (k symbols in reverse order) and an observed next symbol,
%   traverse the trie along the context path. If a node doesn't exist, create it.
%   At the final node reached, increment the count for nextSym.
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