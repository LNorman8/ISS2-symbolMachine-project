%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VARIABLE-CONTEXT MODEL WITH TRIE LOOKUP AND EXPONENTIAL DEPTH WEIGHTING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [bitsPerSymbol, selectedParams] = expmodelTrie(name, varargin)
if nargin < 1 || isempty(name)
    name = 'Hawaiian';
end
if isstring(name)
    name = char(name);
end
if ~ischar(name)
    error('Dataset name must be a character vector or string scalar.');
end

opts.k = [];
opts.weightBase = [];
opts.priorScale = [];
opts.gamma = [];
opts.mixAlpha = [];
opts.shallowK = [];
opts.autoTune = true;
opts.useDatasetTuning = true;
opts.usePerClassTuning = true;
opts.useGlobalDefault = true;
opts.datasetParamsFile = 'datasetParams.mat';
opts.globalDefaultFile = 'globalDefaultParams.mat';
opts.trainFileOverride = '';
opts.testFileOverride = '';
if mod(numel(varargin),2) ~= 0
    error('expmodelTrie requires name/value parameter pairs.');
end
for vi = 1:2:numel(varargin)
    nameArg = varargin{vi};
    valArg = varargin{vi+1};
    if isstring(nameArg)
        nameArg = char(nameArg);
    end
    if ~ischar(nameArg)
        error('Parameter names must be character vectors or string scalars.');
    end
    switch lower(nameArg)
        case 'k'
            opts.k = valArg;
        case 'weightbase'
            opts.weightBase = valArg;
        case 'priorscale'
            opts.priorScale = valArg;
        case 'gamma'
            opts.gamma = valArg;
        case 'mixalpha'
            opts.mixAlpha = valArg;
        case 'shallowk'
            opts.shallowK = valArg;
        case 'autotune'
            opts.autoTune = toLogical(valArg);
        case 'usedatasettuning'
            opts.useDatasetTuning = toLogical(valArg);
        case 'useperclasstuning'
            opts.usePerClassTuning = toLogical(valArg);
        case 'useglobaldefault'
            opts.useGlobalDefault = toLogical(valArg);
        case 'datasetparamsfile'
            opts.datasetParamsFile = char(valArg);
        case 'globaldefaultfile'
            opts.globalDefaultFile = char(valArg);
        case 'trainfileoverride'
            opts.trainFileOverride = char(valArg);
        case 'testfileoverride'
            opts.testFileOverride = char(valArg);
        otherwise
            error('Unknown parameter name: %s', nameArg);
    end
end

trainFile = ['sequence_' name '_train.mat'];
testFile  = ['sequence_' name '_test.mat'];
if ~isempty(opts.trainFileOverride)
    trainFile = opts.trainFileOverride;
end
if ~isempty(opts.testFileOverride)
    testFile = opts.testFileOverride;
end
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

% Try to resolve parameters in precedence order:
% manual overrides -> dataset-specific -> per-class -> global default -> autoTune -> fallback defaults.
[k, weightBase, priorScale, gamma, mixAlpha, shallowK, paramSource] = resolveRuntimeParams( ...
    name, trainSeq, opts, maxK, weightBaseCands, priorScaleCands, gammaCands);
fprintf('Using %s\n', paramSource);

if isempty(mixAlpha)
    mixAlpha = defaultMixAlpha(k);
end
if isempty(shallowK)
    shallowK = defaultShallowK(k);
end
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
% Bootstrap the initial context from the start of the training sequence so
% that the first test predictions already have a full k-symbol context.
% (Now the start of the sequence instead of the end, spec. for lexical data
% where that should hopefully help a bit more)
context = trainSeq(1:k+1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- VARIABLE-LENGTH FORECAST LOOP ---
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exponential depth weighting with node-level JM smoothing. During testing,
% each observed symbol is fed back into the trie so the model keeps
% learning online.
for t = 1:T
    probVec = forecastFromTrieMixture(context, trieChildren, trieCounts, ...
        priorProb, k, weightBase, priorScale, gamma, mixAlpha, shallowK);
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
global SYMBOLDATA
bitsPerSymbol = SYMBOLDATA.totalPenaltyInBits / T;
selectedParams = struct( ...
    'k', k, ...
    'weightBase', weightBase, ...
    'priorScale', priorScale, ...
    'gamma', gamma, ...
    'mixAlpha', mixAlpha, ...
    'shallowK', shallowK, ...
    'source', paramSource);
end

function [k, weightBase, priorScale, gamma, mixAlpha, shallowK, source] = resolveRuntimeParams( ...
    name, trainSeq, opts, maxK, weightBaseCands, priorScaleCands, gammaCands)
k = [];
weightBase = [];
priorScale = [];
gamma = [];
mixAlpha = [];
shallowK = [];
source = 'fallback defaults';

manualOverride = ~isempty(opts.k) || ~isempty(opts.weightBase) || ~isempty(opts.priorScale) || ...
    ~isempty(opts.gamma) || ~isempty(opts.mixAlpha) || ~isempty(opts.shallowK);
if manualOverride
    k = opts.k;
    if isempty(k)
        k = 8;
    end
    weightBase = opts.weightBase;
    if isempty(weightBase)
        weightBase = 3.0;
    end
    priorScale = opts.priorScale;
    if isempty(priorScale)
        priorScale = 0.5;
    end
    gamma = opts.gamma;
    if isempty(gamma)
        gamma = 8.0;
    end
    mixAlpha = opts.mixAlpha;
    shallowK = opts.shallowK;
    source = 'manual overrides';
    return;
end

if opts.useDatasetTuning && isfile(opts.datasetParamsFile)
    try
        data = load(opts.datasetParamsFile);
        if isfield(data, 'datasetParams')
            matchIdx = find(strcmp({data.datasetParams.name}, name), 1);
            if ~isempty(matchIdx)
                params = data.datasetParams(matchIdx).params;
                [k, weightBase, priorScale, gamma, mixAlpha, shallowK] = normalizeParams(params, maxK);
                source = sprintf('dataset-specific tuning for %s', name);
                return;
            end
        end
    catch
        % Fall through to class/global/default resolution.
    end
end

if opts.usePerClassTuning && isfile('perClassParams.mat')
    try
        data = load('perClassParams.mat');
        if isfield(data, 'bestParamsPerClass') && isfield(data, 'classThresholds') && ...
                isfield(data, 'classNames') && isfield(data, 'classMembers')
            entropy = computeEntropy(trainSeq);
            classIdx = [];
            classThresholds = data.classThresholds;
            for c = 1:4
                if entropy >= classThresholds(c) && entropy < classThresholds(c+1)
                    classIdx = c;
                    break;
                end
            end
            if ~isempty(classIdx) && classIdx <= numel(data.bestParamsPerClass) && ...
                    ~isempty(data.bestParamsPerClass{classIdx})
                params = data.bestParamsPerClass{classIdx};
                [k, weightBase, priorScale, gamma, mixAlpha, shallowK] = normalizeParams(params, maxK);
                source = sprintf('per-class tuning (%s)', data.classNames{classIdx});
                return;
            end
        end
    catch
        % Fall through.
    end
end

if opts.useGlobalDefault && isfile(opts.globalDefaultFile)
    try
        data = load(opts.globalDefaultFile);
        if isfield(data, 'globalDefaultParams')
            [k, weightBase, priorScale, gamma, mixAlpha, shallowK] = normalizeParams(data.globalDefaultParams, maxK);
            source = 'global default params';
            return;
        end
    catch
        % Fall through.
    end
end

if opts.autoTune
    [k, weightBase, priorScale, gamma] = chooseAdaptiveParams(trainSeq, ...
        maxK, weightBaseCands, priorScaleCands, gammaCands);
    mixAlpha = defaultMixAlpha(k);
    shallowK = defaultShallowK(k);
    source = 'autoTune training CV';
else
    k = 8;
    weightBase = 3.0;
    priorScale = 0.5;
    gamma = 8.0;
    mixAlpha = defaultMixAlpha(k);
    shallowK = defaultShallowK(k);
    source = 'fallback defaults';
end
end

function [k, weightBase, priorScale, gamma, mixAlpha, shallowK] = normalizeParams(params, maxK)
k = params.k;
if isempty(k)
    k = min(8, maxK);
end
weightBase = params.weightBase;
if isempty(weightBase)
    weightBase = 3.0;
end
priorScale = params.priorScale;
if isempty(priorScale)
    priorScale = 0.5;
end
gamma = params.gamma;
if isempty(gamma)
    gamma = 8.0;
end
if isfield(params, 'mixAlpha') && ~isempty(params.mixAlpha)
    mixAlpha = params.mixAlpha;
else
    mixAlpha = defaultMixAlpha(k);
end
if isfield(params, 'shallowK') && ~isempty(params.shallowK)
    shallowK = params.shallowK;
else
    shallowK = defaultShallowK(k);
end
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

function tf = toLogical(value)
if islogical(value)
    tf = value;
elseif isnumeric(value)
    tf = value ~= 0;
elseif isstring(value)
    tf = any(strcmpi(string(value), ["true","1","yes","on"]));
elseif ischar(value)
    tf = any(strcmpi(string(value), ["true","1","yes","on"]));
else
    error('Expected a logical-compatible value.');
end
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

function entropy = computeEntropy(sequence)
% Compute Shannon entropy of a symbol sequence
% Input: sequence (row vector of integers 1-9)
% Output: entropy in bits
counts = histcounts(sequence, 0.5:1:9.5);
probs = counts / sum(counts);
probs(probs == 0) = [];
entropy = -sum(probs .* log2(probs));
end