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
% Automatic tuning is the default when the caller does not provide all four parameters.
ULTRATUNE = true; % use ranges for each dataset
trainFile = ['sequence_' name '_train.mat'];
testFile  = ['sequence_' name '_test.mat'];
seq_struct = load(trainFile);
sequence = seq_struct.sequence; % Extract the sequence from the loaded data
trainSeq = sequence(:)'; % Ensure row vector
N = length(trainSeq);
% Test sequence length is needed up front so the trie can reserve room for
% online updates while forecasting.
T = initializeSymbolMachineS26(testFile, 0);

% PARAMETER TUNING:
%   Uses inner cross-validation loop (70% train / 30% validation split on training data)
%   to search for best (k, weightBase, priorScale, gamma). Evaluates each candidate
%   by building a trie on train part, then computing bits-per-symbol on val part.
%   Dataset-specific candidate grids (ULTRATUNE=true) vs. wide general defaults.
%
%   - k: Max context depth. Trade-off: deeper = more context but more overfitting risk.
%   - weightBase: Exponential base for depth weighting. >1 emphasizes deeper contexts;
%     =1 means all depths equally weighted.
%   - priorScale: Scales the global prior contribution. High values make model rely
%     more on overall symbol frequency (useful when data is sparse).
%   - gamma: Jelinek-Mercer smoothing. Higher values interpolate empirical counts
%     toward the prior, reducing overfitting on low-count nodes.
%
if userProvidedParams
    % Use user-provided values exactly as entered.
    paramSource = 'USER-PROVIDED';
else
    % Candidate grids for depth weighting and smoothing.
    % Start with a strong general grid biased toward deeper contexts, then
    % override per dataset when a tailored search range is available.
    maxK = min(12, max(1, N - 1));
    weightBaseCands = [1, 1.001, 1.05, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 5.0, 20.0];
    priorScaleCands = [0, 0.01, 0.1, 0.2, 0.5, 1.0];
    gammaCands = [0, 1, 1.05, 1.1, 1.5, 4.0, 8.0, 16.0, 31, 32.0, 33];

    if ULTRATUNE
        switch name
            case 'Hawaiian' % Bias toward higher-k contexts; this dataset benefits from deeper history.
                maxK = min(12, max(1, N - 1));
                weightBaseCands = [1.00, 1.05, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20, 1.22, 1.24, 1.25, 1.26, 1.28, 1.30];
                priorScaleCands = [0.25, 0.50, 0.75, 1.00];
                gammaCands = [0, 0.1, 0.25, 0.5, 1];
            % case 'Dickens' % Allow deeper contexts than before; keep the tuned smoothing grid.
            %     maxK = min(12, max(1, N - 1));
            %     weightBaseCands = 1.8:0.05:2.4;
            %     priorScaleCands = 0.05:0.05:0.20;
            %     gammaCands = 1:0.02:2.4;
                k = 9;
                weightBase = 2;
                priorScale = 0.10;
                gamma = 2;
            case 'DIAtemp'
                maxK = min(10, max(1, N - 1));
                weightBaseCands = [1.00, 1.01, 1.03, 1.05, 1.08, 1.10, 1.12];
                priorScaleCands = [0.00, 0.01, 0.02, 0.05, 0.10];
                gammaCands = [0, 0.25, 0.5, 0.75, 1.0];
            case 'DIAwind'
                maxK = min(10, max(1, N - 1));
                weightBaseCands = [2.0:0.1:3.0];
                priorScaleCands = [0.2:0.05:0.80];
                gammaCands = [0:2:64];
            case 'ElecDemand'
                % maxK = min(12, max(1, N - 1));
                % weightBaseCands = [1.00, 1.01, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.15];
                % priorScaleCands = [0.00, 0.01, 0.05, 0.10, 0.15, 0.20];
                % gammaCands = [0, 0.5, 1];
                k = 12;
                weightBase = 1.00;
                priorScale = 0.01;
                gamma = 0;
            case 'HoustonRain'
                maxK = min(12, max(1, N - 1));
                weightBaseCands = [1.12, 1.15, 1.18, 1.20, 1.22, 1.25, 1.28];
                priorScaleCands = [0.50, 0.75, 1.00, 1.25, 1.50];
                gammaCands = [12, 16, 20, 24, 28, 32, 40];
            case 'solarWind'
                maxK = min(10, max(1, N - 1));
                weightBaseCands = [1.00, 1.02, 1.04, 1.06, 1.08, 1.12, 1.15, 1.20];
                priorScaleCands = [0.00, 0.01, 0.03, 0.05, 0.08, 0.10, 0.15];
                gammaCands = [0, 0.1, 0.25, 0.5, 1, 2];
        end
    end

    if isnan(k) % Allow handtuned values to go untouched (since k isn't a list so won't survive tuning)
    [k, weightBase, priorScale, gamma] = chooseAdaptiveParams(trainSeq, ...
        maxK, weightBaseCands, priorScaleCands, gammaCands);
    end
    paramSource = 'TUNED';
end

% Display chosen params and their source
fprintf('Chosen params for %s [%s]: k=%d, weightBase=%.2f, priorScale=%.2f, gamma=%.2f\n', ...
    name, paramSource, k, weightBase, priorScale, gamma);
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
% TRIE DATA STRUCTURE:
%   - trieChildren(node, sym) = child node index (0 = no child)
%   - trieCounts(node, sym)   = count of next symbol sym observed after context
%
%   Root node 1 = empty context (depth 0); priorProb is used directly, no counts.
%
%   KEY INSIGHT: Context is stored in REVERSE order (most-recent symbol first).
%   A node reached by traversing s1→s2→...→sd from root represents the context
%   [recent, ..., older] = [s1, s2, ..., sd], where s1 was just observed.
%   In forecastFromTrie, we walk context(end-d+1), which extracts symbols in
%   reverse order to match this trie structure.
%
%   Example: if context = [3 1 4] and k=3, then:
%     - context(end) = 4 (most recent, depth 1)
%     - context(end-1) = 1 (depth 2)
%     - context(end-2) = 3 (depth 3, oldest)
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- VARIABLE-LENGTH FORECAST LOOP WITH ONLINE ADAPTATION ---
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For each test symbol:
%   1. Walk the trie using current context, accumulating weighted probabilities
%      from each depth level (weighted by weightBase^depth).
%   2. At each trie node, use Jelinek-Mercer smoothing (controlled by gamma)
%      to interpolate between empirical counts and global prior.
%   3. Query Symbol Machine with computed probability vector.
%   4. Observe the true symbol and immediately update both the global prior
%      and trie counts (online learning). This allows model to adapt to
%      any shift in test set distribution.
%   5. Slide context window: drop oldest symbol, prepend newly observed symbol.
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
%
% INNER CROSS-VALIDATION FOR HYPERPARAMETER TUNING:
%   Uses a tail window of training data (last 12k symbols or all if smaller) to
%   keep evaluation tractable for large datasets. Splits this window 70/30 into
%   a train part and validation part. For each candidate (k, weightBase, priorScale, gamma):
%     - Builds a trie on the train part
%     - Forecasts on validation part, computing bits-per-symbol loss
%     - Updates global prior online as validation symbols arrive
%   Returns the (k, weightBase, priorScale, gamma) tuple with lowest validation loss.
%
%   Early exit: if trainSeq is very small (<= 40 symbols), returns defaults to avoid
%   overfitting during tuning itself.
%
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
                if bitsPerSymbol < bestBits - 1e-9 || (abs(bitsPerSymbol - bestBits) <= 1e-9 && kCand > kBest)
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
%
% PREDICT NEXT SYMBOL BY WALKING TRIE AND ACCUMULATING WEIGHTED PROBABILITIES:
%   - Start with priorProb scaled by priorScale (depth-0 base).
%   - For each depth d = 1 to k, navigate trie using context(end - d + 1)
%     (extracts symbols in reverse order to match trie's reverse-context indexing).
%   - At each node, extract symbol counts and apply Jelinek-Mercer smoothing:
%     lambda = total_count / (total_count + gamma)
%     depthProb = lambda * empirical + (1 - lambda) * priorProb
%   - Accumulate weighted probability: (weightBase^d) * depthProb
%     Higher weightBase emphasizes deeper (more specific) contexts.
%   - Defensive checks: clamp negative/NaN/Inf values, handle zero-sum cases.
%   - Normalize final accumulated probability vector to sum to 1.
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
%   Used both during training (to build trie from training sequence) and during
%   test forecasting (for online adaptation as symbols are observed).
%
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