%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FORECAST MODEL COMPARISON SCRIPT (S26)
%%% Compares:
%%%   (0) Uniform baseline
%%%   (1) 2nd-order Markov Chain (MC-2)
%%%   (2) 2nd-order Autoregressive (AR-2)
%%%   (3) Adaptive Expert Mixture (AEM) [non-MC/non-AR]
%%% Colorado School of Mines EENG311 - Spring 2026 - Mike Wakin
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc;

%% ================================================================
%% USER SETTINGS: CHANGE DATASET HERE
%% ================================================================
name = 'DIAtemp';
trainFile = ['sequence_' name '_train.mat'];
testFile  = ['sequence_' name '_test.mat'];

% Verbose Symbol Machine output? (0/1)
verboseSM = 0;

% Reproducibility for any random tie behavior, etc.
rng(0);

%% ================================================================
%% LOAD TRAINING DATA (allowed)
%% ================================================================
load(trainFile); % expects variable "sequence"
trainSeq = sequence(:);
if any(trainSeq < 1) || any(trainSeq > 9) || any(abs(trainSeq-round(trainSeq))>0)
    error('Training data must contain integer symbols 1..9 in variable "sequence".');
end

%% ================================================================
%% RUN ALL MODELS
%% ================================================================
results = struct('name',{},'bitsPerSymbol',{},'totalBits',{},'numSymbols',{});

results(end+1) = runUniformModel(testFile,verboseSM);
results(end+1) = runMC2Model(trainSeq,testFile,verboseSM);
results(end+1) = runAR4Model(trainSeq,testFile,verboseSM);
results(end+1) = runAdaptiveExpertModel(trainSeq,testFile,verboseSM);

%% ================================================================
%% DISPLAY COMPARISON TABLE
%% ================================================================
fprintf('\n============================================================\n');
fprintf('Dataset comparison on TEST file: %s\n', testFile);
fprintf('Training file used: %s\n', trainFile);
fprintf('Lower bits/symbol is better.\n');
fprintf('============================================================\n');

% Sort by performance
allBPS = [results.bitsPerSymbol];
[~,ord] = sort(allBPS,'ascend');

fprintf('%-30s | %12s | %12s | %10s\n','Model','bits/symbol','total bits','N');
fprintf('%s\n',repmat('-',1,74));
for i = 1:length(ord)
    r = results(ord(i));
    fprintf('%-30s | %12.6f | %12.3f | %10d\n',r.name,r.bitsPerSymbol,r.totalBits,r.numSymbols);
end
fprintf('%s\n\n',repmat('=',1,74));

%% ================================================================
%% LOCAL MODEL RUNNERS
%% ================================================================

function out = runUniformModel(testFile,verboseSM)
    fprintf('\n--- Running Uniform Baseline ---\n');
    N = initializeSymbolMachineS26(testFile,verboseSM);
    probs = ones(1,9)/9;
    for t = 1:N
        [~,~] = symbolMachineS26(probs);
    end
    stats = collectSMStats();
    reportSymbolMachineS26;
    out.name = 'Uniform';
    out.bitsPerSymbol = stats.bitsPerSymbol;
    out.totalBits = stats.totalBits;
    out.numSymbols = stats.numSymbols;
end

function out = runMC2Model(trainSeq,testFile,verboseSM)
    fprintf('\n--- Running 2nd-Order Markov Chain (MC-2) ---\n');

    % Train 2nd-order MC with Laplace smoothing
    counts = ones(9,9,9); % past2, past1, current
    Ntr = length(trainSeq);
    for t = 3:Ntr
        p2 = trainSeq(t-2);
        p1 = trainSeq(t-1);
        c  = trainSeq(t);
        counts(p2,p1,c) = counts(p2,p1,c) + 1;
    end
    probM = counts;
    for p2 = 1:9
        for p1 = 1:9
            row = squeeze(probM(p2,p1,:));
            row = row / sum(row);
            probM(p2,p1,:) = row;
        end
    end

    % Test via Symbol Machine
    N = initializeSymbolMachineS26(testFile,verboseSM);
    probs = ones(1,9)/9;
    [past2,~] = symbolMachineS26(probs);
    [past1,~] = symbolMachineS26(probs);

    for t = 3:N
        probs = squeeze(probM(past2,past1,:)).';
        [sym,~] = symbolMachineS26(probs);
        past2 = past1;
        past1 = sym;
    end

    stats = collectSMStats();
    reportSymbolMachineS26;
    out.name = 'Markov Chain (order 2)';
    out.bitsPerSymbol = stats.bitsPerSymbol;
    out.totalBits = stats.totalBits;
    out.numSymbols = stats.numSymbols;
end

function out = runAR4Model(trainSeq,testFile,verboseSM)
    fprintf('\n--- Running 4nd-Order Autoregressive (AR-4) ---\n');

    % Train AR(4) model
    Mdl = arima(4,0,0);
    Mdl = estimate(Mdl,trainSeq,'Display','off');

    % Test via Symbol Machine
    N = initializeSymbolMachineS26(testFile,verboseSM);
    probs = ones(1,9)/9;
    [past4,~] = symbolMachineS26(probs);
    [past3,~] = symbolMachineS26(probs);
    [past2,~] = symbolMachineS26(probs);
    [past1,~] = symbolMachineS26(probs);
    for t = 5:N
        [mu,~,V] = forecast(Mdl,1,[past4;past3;past2;past1]);
        sigma = sqrt(max(V,1e-12));

        % Convert Gaussian forecast to 9-symbol pmf
        x = (1:9)';
        p = normpdf(x,mu,sigma);
        p = max(p,1e-12);
        p = p/sum(p);

        [sym,~] = symbolMachineS26(p.');
         past4 = past3;
         past3 = past2;
         past2 = past1;
         past1 = sym;
    end

    stats = collectSMStats();
    reportSymbolMachineS26;
    out.name = 'Autoregressive (order 4)';
    out.bitsPerSymbol = stats.bitsPerSymbol;
    out.totalBits = stats.totalBits;
    out.numSymbols = stats.numSymbols;
end

function out = runAdaptiveExpertModel(trainSeq,testFile,verboseSM)
    fprintf('\n--- Running Adaptive Expert Mixture (AEM) ---\n');

    % ---------------- Hyperparameters ----------------
    Wshort   = 25;
    Wtrend   = 12;
    Wcal     = 80;
    eta      = 0.45;
    lambda   = 0.01;
    alphaLap = 0.5;
    epsProb  = 1e-12;
    Tmin     = 0.75;
    Tmax     = 1.35;
    K        = 9;
    Lctx     = 6;

    % ---------------- Train expert statistics ----------------
    Ntr = length(trainSeq);

    % Expert 1 (global PMF)
    globalCounts = alphaLap*ones(9,1);
    for i = 1:Ntr
        globalCounts(trainSeq(i)) = globalCounts(trainSeq(i)) + 1;
    end
    globalPMF = globalCounts/sum(globalCounts);

    % Expert 3 (run length)
    runCont = ones(5,1);
    runStop = ones(5,1);
    r = 1;
    while r <= Ntr
        s = trainSeq(r);
        j = r+1;
        while j<=Ntr && trainSeq(j)==s
            j = j+1;
        end
        runLen = j-r;
        b = min(runLen,5);
        if runLen >= 2
            runCont(b) = runCont(b) + (runLen-1);
        end
        if j <= Ntr
            runStop(b) = runStop(b) + 1;
        end
        r = j;
    end
    pContinueByBucket = runCont./(runCont+runStop);

    % Expert 5 (periodicity: 2..8)
    periodList = 2:8;
    numPeriods = length(periodList);
    periodCounts = cell(numPeriods,1);
    for ip = 1:numPeriods
        pLen = periodList(ip);
        C = alphaLap*ones(pLen,9);
        for t = 1:Ntr
            ph = mod(t-1,pLen)+1;
            C(ph,trainSeq(t)) = C(ph,trainSeq(t)) + 1;
        end
        periodCounts{ip} = C;
    end

    % Expert 6 (trend-bin)
    trendCounts = alphaLap*ones(5,9);
    for t = 3:Ntr
        d = trainSeq(t-1)-trainSeq(t-2);
        b = trendBin(d);
        trendCounts(b,trainSeq(t)) = trendCounts(b,trainSeq(t)) + 1;
    end

    % Expert 7 (volatility-state)
    volCounts = alphaLap*ones(3,9);
    for t = max(3,Wtrend+1):Ntr
        recent = trainSeq(t-Wtrend:t-1);
        sdev = std(double(recent));
        v = volatilityState(sdev);
        volCounts(v,trainSeq(t)) = volCounts(v,trainSeq(t)) + 1;
    end

    % Expert 8 (nearest-pattern memory)
    ctxMatrix = [];
    ctxNext   = [];
    for t = Lctx+1:Ntr
        ctxMatrix = [ctxMatrix; trainSeq(t-Lctx:t-1).']; %#ok<AGROW>
        ctxNext   = [ctxNext; trainSeq(t)]; %#ok<AGROW>
    end

    % ---------------- Test phase ----------------
    N = initializeSymbolMachineS26(testFile,verboseSM);

    w = ones(K,1)/K;
    probs = ones(1,9)/9;
    [h1,~] = symbolMachineS26(probs);
    [h2,~] = symbolMachineS26(probs);
    history = [h1; h2];
    recentPen = [];

    for t = 3:N
        P = zeros(9,K);
        P(:,1) = safeNormalize(globalPMF,epsProb);
        P(:,2) = expertShortWindow(history,Wshort,alphaLap,epsProb);
        P(:,3) = expertRunLength(history,pContinueByBucket,alphaLap,epsProb);
        P(:,4) = expertRecency(history,epsProb);
        P(:,5) = expertPeriodicity(t,periodList,periodCounts,epsProb);
        P(:,6) = expertTrend(history,trendCounts,epsProb);
        P(:,7) = expertVolatility(history,volCounts,Wtrend,epsProb);
        P(:,8) = expertNearestPattern(history,ctxMatrix,ctxNext,Lctx,alphaLap,epsProb);
        P(:,9) = ones(9,1)/9;

        p = P*w;
        p = safeNormalize(p,epsProb);

        T = temperatureFromRecentPenalties(recentPen,Wcal,Tmin,Tmax);
        p = p.^(1/T);
        p = safeNormalize(p,epsProb);

        [sym,pen] = symbolMachineS26(p.');

        pkTrue = max(P(sym,:)',epsProb);
        w = w .* exp(eta*log(pkTrue));
        w = w/sum(w);
        w = (1-lambda)*w + lambda*(ones(K,1)/K);
        w = w/sum(w);

        recentPen = [recentPen; pen]; %#ok<AGROW>
        if length(recentPen)>Wcal
            recentPen = recentPen(end-Wcal+1:end);
        end

        history = [history; sym]; %#ok<AGROW>
    end

    stats = collectSMStats();
    reportSymbolMachineS26;
    out.name = 'Adaptive Expert Mixture';
    out.bitsPerSymbol = stats.bitsPerSymbol;
    out.totalBits = stats.totalBits;
    out.numSymbols = stats.numSymbols;
end

%% ================================================================
%% SHARED UTILITIES
%% ================================================================

function s = collectSMStats()
    global SYMBOLDATA
    s.totalBits = SYMBOLDATA.totalPenaltyInBits;
    s.numSymbols = SYMBOLDATA.sequenceLength;
    s.bitsPerSymbol = s.totalBits/s.numSymbols;
end

function p = safeNormalize(x,epsProb)
    x = x(:);
    x = max(x,epsProb);
    p = x/sum(x);
end

function b = trendBin(d)
    if d <= -2
        b = 1;
    elseif d == -1
        b = 2;
    elseif d == 0
        b = 3;
    elseif d == 1
        b = 4;
    else
        b = 5;
    end
end

function v = volatilityState(sdev)
    if sdev < 1.0
        v = 1;   % low
    elseif sdev < 2.0
        v = 2;   % medium
    else
        v = 3;   % high
    end
end

function p = expertShortWindow(history,Wshort,alphaLap,epsProb)
    counts = alphaLap*ones(9,1);
    n = length(history);
    a = max(1,n-Wshort+1);
    for i = a:n
        counts(history(i)) = counts(history(i)) + 1;
    end
    p = safeNormalize(counts,epsProb);
end

function p = expertRunLength(history,pContinueByBucket,alphaLap,epsProb)
    last = history(end);
    runLen = 1;
    i = length(history)-1;
    while i>=1 && history(i)==last
        runLen = runLen + 1;
        i = i-1;
    end
    b = min(runLen,5);
    pCont = pContinueByBucket(b);

    counts = alphaLap*ones(9,1);
    switchPMF = counts/sum(counts);
    switchPMF(last) = 0;
    switchPMF = safeNormalize(switchPMF,epsProb);

    p = (1-pCont)*switchPMF;
    p(last) = p(last) + pCont;
    p = safeNormalize(p,epsProb);
end

function p = expertRecency(history,epsProb)
    n = length(history);
    scores = zeros(9,1);
    for s = 1:9
        idx = find(history==s,1,'last');
        if isempty(idx)
            scores(s) = 1;
        else
            gap = n-idx+1;
            scores(s) = 1/gap;
        end
    end
    p = safeNormalize(scores,epsProb);
end

function p = expertPeriodicity(t,periodList,periodCounts,epsProb)
    numPeriods = length(periodList);
    mix = zeros(9,1);
    wPer = ones(numPeriods,1)/numPeriods;
    for ip = 1:numPeriods
        pLen = periodList(ip);
        ph = mod(t-1,pLen)+1;
        C = periodCounts{ip};
        row = C(ph,:).';
        row = safeNormalize(row,epsProb);
        mix = mix + wPer(ip)*row;
    end
    p = safeNormalize(mix,epsProb);
end

function p = expertTrend(history,trendCounts,epsProb)
    if length(history)<2
        p = ones(9,1)/9;
        return;
    end
    d = history(end)-history(end-1);
    b = trendBin(d);
    p = safeNormalize(trendCounts(b,:).',epsProb);
end

function p = expertVolatility(history,volCounts,Wtrend,epsProb)
    n = length(history);
    if n < max(3,Wtrend)
        p = ones(9,1)/9;
        return;
    end
    recent = history(max(1,n-Wtrend+1):n);
    sdev = std(double(recent));
    v = volatilityState(sdev);
    p = safeNormalize(volCounts(v,:).',epsProb);
end

function p = expertNearestPattern(history,ctxMatrix,ctxNext,Lctx,alphaLap,epsProb)
    if length(history)<Lctx || isempty(ctxMatrix)
        p = ones(9,1)/9;
        return;
    end
    q = double(history(end-Lctx+1:end)).';
    D = sum(abs(double(ctxMatrix)-q),2);
    [Ds,idx] = sort(D,'ascend');
    Knear = min(40,length(idx));
    counts = alphaLap*ones(9,1);
    sigma = max(1,median(Ds(1:Knear))+epsProb);

    for k = 1:Knear
        j = idx(k);
        wt = exp(-(Ds(k)^2)/(2*sigma^2));
        s = ctxNext(j);
        counts(s) = counts(s) + wt;
    end
    p = safeNormalize(counts,epsProb);
end

function T = temperatureFromRecentPenalties(recentPen,Wcal,Tmin,Tmax)
    if isempty(recentPen)
        T = 1;
        return;
    end
    m = mean(recentPen(max(1,end-Wcal+1):end));
    ref = 2.5;       % rough center (uniform for 9 symbols is 3.1699)
    delta = m-ref;
    T = 1 + 0.12*delta;
    T = max(Tmin,min(Tmax,T));
end