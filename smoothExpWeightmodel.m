%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VARIABLE-CONTEXT MODEL WITH TRIE LOOKUP AND EXPONENTIAL DEPTH WEIGHTING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = smoothExpWeightmodel(name)
arguments
    name (1,:) char = 'Hawaiian'
end
    trainFile = ['sequence_' name '_train.mat'];
    testFile  = ['sequence_' name '_test.mat'];
    seq_struct = load(trainFile);
    sequence = seq_struct.sequence; % Extract the sequence from the loaded data
    trainSeq = sequence(:)'; % Ensure row vector
    N = length(trainSeq);
    % ============================================================
    % 1. Adaptive k: choose context depth based on training size.
    %    Target ~N/9^k >= 10 training samples per length-k context.
    % ============================================================
    k = max(1, min(8, floor(log(N / 10) / log(9))));
    % ============================================================
    % 2. Precompute global prior (marginal distribution)
    % ============================================================
    priorCounts = zeros(1, 9);
    for sym = 1:9
        priorCounts(sym) = sum(trainSeq == sym);
    end
    priorProb = priorCounts / sum(priorCounts);
    % ============================================================
    % 3. Build suffix trie over training sequence
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
    % Bootstrap the initial context with the end of the training data so that
    % the first test predictions already have a meaningful k-symbol context.
    context = trainSeq(end-k+1:end);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % --- VARIABLE-LENGTH FORECAST LOOP ---
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Exponential depth weighting: each depth d contributes base^d raw counts
    % to an accumulator, so deeper (more specific) contexts dominate the
    % final distribution while shallower contexts still provide a smoothing
    % floor.  The unigram prior (scaled by 0.5) acts as the depth-0 baseline.
    % This mirrors the original expWeightmodel's weight = 3^L scheme but
    % uses the O(k) trie lookup instead of an O(N*k) training-set scan.
    base = 3; % exponential weight base (depth d gets base^d times the raw counts)
    for t = 1:T
        % Start with a small unigram baseline (same role as priorProb * 0.5
        % in the original expWeightmodel).
        accumCounts = priorProb * 0.5;
        node = uint32(1); % root
        for d = 1:k
            sym = context(end - d + 1); % d-th most recent symbol
            childNode = trieChildren(node, sym);
            if childNode == 0
                break; % context unseen at this depth; stop early
            end
            node = childNode;
            depthCounts = trieCounts(node, :);
            % Add exponentially weighted raw counts: deeper depth = much
            % higher weight, so a well-attested deep context dominates.
            accumCounts = accumCounts + (base^d) * depthCounts;
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