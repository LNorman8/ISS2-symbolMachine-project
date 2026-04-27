%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EXPONENTIALLY-WEIGHTED VARIABLE-CONTEXT MODEL (NON-MC / NON-AR) (BASELINE! DO NOT MODIFY)
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
    k = 8; % Max window length (increased to 8 to catch deep structures)
    % ============================================================
    % 1. Precompute global prior (marginal distribution)
    % ============================================================
    priorCounts = zeros(1,9);
    for sym = 1:9
        priorCounts(sym) = sum(trainSeq == sym);
    end
    priorProb = priorCounts / sum(priorCounts);
    % ============================================================
    % 2. Precompute training windows for extremely fast vectorization
    % ============================================================
    % X_train holds the contexts, Y_train holds the target next symbols
    X_train = zeros(N-k, k);
    Y_train = zeros(N-k, 1);
    for i = 1:N-k
        X_train(i, :) = trainSeq(i:i+k-1);
        Y_train(i) = trainSeq(i+k);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % --- TEST SET SETUP (Symbol Machine) ---
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    T = initializeSymbolMachineS26(testFile,0);
    % Bootstrap initial context using the empirical prior (better than uniform)
    context = zeros(1,k);
    for i = 1:k
        [symbol,penalty] = symbolMachineS26(priorProb);
        context(i) = symbol;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % --- VARIABLE-LENGTH FORECAST LOOP ---
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for t = k+1:T
        query = context;
        % Start with a weak baseline heavily informed by global training frequencies
        % This fundamentally replaces naive ones(1,9) smoothing
        nextCounts = priorProb * 0.5;
        % Check all suffix lengths from 1 up to max k
        % Stack their probability profiles together
        for L = 1:k
            subQuery = query(end-L+1:end);
            subX = X_train(:, end-L+1:end);
            % Vectorized exact match for length L
            % 'all(..., 2)' checks if the whole row matches perfectly
            matches = all(subX == subQuery, 2);
            if any(matches)
                matchedNextSyms = Y_train(matches);
                % Exponentially higher reward for deeper context matches
                weight = 3^L;
                for sym = 1:9
                    counts = sum(matchedNextSyms == sym);
                    nextCounts(sym) = nextCounts(sym) + counts * weight;
                end
            end
        end
        % ============================================================
        % Convert to probability distribution
        % ============================================================
        probVec = nextCounts / sum(nextCounts);
        probVec = probVec(:)'; % ensure row vector (1x9)
        % ============================================================
        % Symbol Machine step
        % ============================================================
        [symbol,penalty] = symbolMachineS26(probVec);
        % ============================================================
        % Update context (sliding window)
        % ============================================================
        context = [context(2:end), symbol];
    end
    reportSymbolMachineS26;
end