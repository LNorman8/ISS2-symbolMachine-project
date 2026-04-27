%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VARIABLE-CONTEXT MODEL WITH TRIE LOOKUP AND JELINEK-MERCER SMOOTHING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [] = expWeightmodel(name)
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
    % Bootstrap the initial context with the unigram prior.
    % The first k symbols incur penalty at the unigram rate (unavoidable
    % without any prior context).
    context = zeros(1, k);
    for i = 1:k
        % symbolMachineS26 accumulates penalty internally into SYMBOLDATA;
        % the local return value is not needed here.
        [symbol, ~] = symbolMachineS26(priorProb);
        context(i) = symbol;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % --- VARIABLE-LENGTH FORECAST LOOP ---
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Jelinek-Mercer interpolation weight lambda.
    % At each depth d: P_d = lambda * P_ML_d + (1-lambda) * P_{d-1}
    % Starting from P_0 = priorProb (unigram).
    lambda = 0.5;
    for t = k + 1:T
        if mod(t, 1000) == 0
            disp(t);
        end
        % Walk the trie from root using the current context (most-recent
        % symbol first).  At each depth, blend the local ML estimate with
        % the accumulated shallower estimate via Jelinek-Mercer smoothing.
        probVec = priorProb; % depth-0 base: unigram
        node = uint32(1);   % root
        for d = 1:k
            sym = context(end - d + 1); % d-th most recent symbol
            childNode = trieChildren(node, sym);
            if childNode == 0
                break; % context unseen at this depth; stop early
            end
            node = childNode;
            depthCounts = trieCounts(node, :);
            total = sum(depthCounts);
            if total > 0
                P_ML = depthCounts / total;
                probVec = lambda * P_ML + (1 - lambda) * probVec;
            end
        end
        % ============================================================
        % Ensure valid probability vector (row, non-negative, sums to 1)
        % ============================================================
        probVec = max(probVec(:)', 1e-12);
        probVec = probVec / sum(probVec);
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