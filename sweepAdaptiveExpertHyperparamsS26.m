%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% HYPERPARAMETER SWEEP FOR ADAPTIVE EXPERT MIXTURE (S26)
%%% Non-MC / Non-AR model tuning using TRAIN/VALID split only.
%%% Colorado School of Mines EENG311 - Spring 2026 - Mike Wakin
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc;
rng(0);

%% ================================================================
%% USER SETTINGS
%% ================================================================
name = 'DIAtemp';
trainFile = ['sequence_' name '_train.mat'];
testFile  = ['sequence_' name '_test.mat'];

% Fraction of TRAINING sequence used for sub-train (rest = validation)
subTrainFrac = 0.70;

% If true, runs a smaller sweep for quick debugging
fastMode = false;

% Optional: after sweep, evaluate best config on official test set
runBestOnOfficialTest = true;
verboseSM = 0;

%% ================================================================
%% LOAD TRAINING DATA
%% ================================================================
load(trainFile); % expects variable "sequence"
fullSeq = sequence(:);
N = length(fullSeq);

if any(fullSeq < 1) || any(fullSeq > 9) || any(abs(fullSeq-round(fullSeq))>0)
    error('Sequence must contain integer symbols 1..9 in variable "sequence".');
end
if N < 200
    error('Training sequence too short for reliable sweep.');
end

splitIdx = max(50, min(N-50, floor(subTrainFrac*N)));
subTrainSeq = fullSeq(1:splitIdx);
valSeq      = fullSeq(splitIdx+1:end);

fprintf('Loaded %s with %d symbols.\n',trainFile,N);
fprintf('Sub-train: %d symbols | Validation: %d symbols\n',length(subTrainSeq),length(valSeq));

%% ================================================================
%% DEFINE HYPERPARAMETER GRID
%% ================================================================
if fastMode
    grid.Wshort   = [15 35];
    grid.Wtrend   = [8 16];
    grid.Wcal     = [40 100];
    grid.eta      = [0.25 0.55];
    grid.lambda   = [0.005 0.02];
    grid.alphaLap = [0.3 1.0];
    grid.Lctx     = [4 8];
    grid.Tmin     = [0.80];
    grid.Tmax     = [1.25];
    grid.Knear    = [20 40];
else
    grid.Wshort   = [10 20 35 60];
    grid.Wtrend   = [6 12 20];
    grid.Wcal     = [30 80 160];
    grid.eta      = [0.15 0.30 0.45 0.65];
    grid.lambda   = [0.002 0.01 0.03];
    grid.alphaLap = [0.1 0.5 1.0];
    grid.Lctx     = [4 6 8];
    grid.Tmin     = [0.75 0.85];
    grid.Tmax     = [1.25 1.35];
    grid.Knear    = [15 30 50];
end

combos = makeCombos(grid);
numCombos = length(combos);

fprintf('Total hyperparameter combinations: %d\n',numCombos);

%% ================================================================
%% SWEEP
%% ================================================================
results(numCombos) = struct( ...
    'id',[], ...
    'bitsPerSymbol',[], ...
    'totalBits',[], ...
    'numSymbols',[], ...
    'cfg',[] );

tic;
for i = 1:numCombos
    cfg = combos(i);

    [bps,totalBits,numSym] = evaluateAdaptiveExpertOnHoldout(subTrainSeq,valSeq,cfg);

    results(i).id = i;
    results(i).bitsPerSymbol = bps;
    results(i).totalBits = totalBits;
    results(i).numSymbols = numSym;
    results(i).cfg = cfg;

    if mod(i, max(1,floor(numCombos/20)))==0 || i==1 || i==numCombos
        fprintf('Progress %4d/%4d | val bits/symbol = %.6f\n',i,numCombos,bps);
    end
end
elapsed = toc;
fprintf('Sweep completed in %.2f seconds.\n',elapsed);

%% ================================================================
%% RANK + REPORT TOP CONFIGS
%% ================================================================
allBPS = [results.bitsPerSymbol];
[~,ord] = sort(allBPS,'ascend');

topK = min(15,numCombos);
fprintf('\n============================================================\n');
fprintf('Top %d configs by validation bits/symbol (lower is better)\n',topK);
fprintf('============================================================\n');

for r = 1:topK
    idx = ord(r);
    cfg = results(idx).cfg;
    fprintf(['#%2d | ID %4d | bps %.6f | Wshort=%d Wtrend=%d Wcal=%d ' ...
             '| eta=%.3f lambda=%.4f alpha=%.3f Lctx=%d Tmin=%.2f Tmax=%.2f Knear=%d\n'], ...
            r, results(idx).id, results(idx).bitsPerSymbol, ...
            cfg.Wshort, cfg.Wtrend, cfg.Wcal, cfg.eta, cfg.lambda, ...
            cfg.alphaLap, cfg.Lctx, cfg.Tmin, cfg.Tmax, cfg.Knear);
end
fprintf('============================================================\n\n');

bestCfg = results(ord(1)).cfg;
bestValBPS = results(ord(1)).bitsPerSymbol;
fprintf('BEST validation bits/symbol: %.6f\n',bestValBPS);

%% ================================================================
%% OPTIONAL: RUN BEST CONFIG ON OFFICIAL TEST SET (Symbol Machine)
%% ================================================================
if runBestOnOfficialTest
    fprintf('\nRunning best config on official test set: %s\n',testFile);

    % Train expert stats on FULL training sequence
    state = trainAdaptiveExpertState(fullSeq,bestCfg);

    % Evaluate on Symbol Machine test sequence
    [testBPS,testTotal,testN] = evaluateAdaptiveExpertWithSymbolMachine(state,testFile,bestCfg,verboseSM);

    fprintf('\nBest-config OFFICIAL TEST performance:\n');
    fprintf('  bits/symbol = %.6f\n',testBPS);
    fprintf('  total bits  = %.3f\n',testTotal);
    fprintf('  N symbols   = %d\n\n',testN);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function combos = makeCombos(grid)
    combos = struct('Wshort',{},'Wtrend',{},'Wcal',{},'eta',{},'lambda',{}, ...
                    'alphaLap',{},'Lctx',{},'Tmin',{},'Tmax',{},'Knear',{});
    c = 0;
    for a = 1:length(grid.Wshort)
    for b = 1:length(grid.Wtrend)
    for d = 1:length(grid.Wcal)
    for e = 1:length(grid.eta)
    for f = 1:length(grid.lambda)
    for g = 1:length(grid.alphaLap)
    for h = 1:length(grid.Lctx)
    for j = 1:length(grid.Tmin)
    for k = 1:length(grid.Tmax)
    for m = 1:length(grid.Knear)
        if grid.Tmax(k) < grid.Tmin(j)
            continue;
        end
        c = c + 1;
        combos(c).Wshort   = grid.Wshort(a);
        combos(c).Wtrend   = grid.Wtrend(b);
        combos(c).Wcal     = grid.Wcal(d);
        combos(c).eta      = grid.eta(e);
        combos(c).lambda   = grid.lambda(f);
        combos(c).alphaLap = grid.alphaLap(g);
        combos(c).Lctx     = grid.Lctx(h);
        combos(c).Tmin     = grid.Tmin(j);
        combos(c).Tmax     = grid.Tmax(k);
        combos(c).Knear    = grid.Knear(m);
    end
    end
    end
    end
    end
    end
    end
    end
    end
    end
end

function [bps,totalBits,numSym] = evaluateAdaptiveExpertOnHoldout(trainSeq,valSeq,cfg)
    epsProb = 1e-12;
    K = 9;

    % Train state on sub-train
    state = trainAdaptiveExpertState(trainSeq,cfg);

    % Initialize online phase with first two validation symbols
    if length(valSeq) < 3
        error('Validation sequence too short.');
    end
    w = ones(K,1)/K;
    history = [valSeq(1); valSeq(2)];
    totalBits = -log2(1/9) - log2(1/9); % first two use uniform
    recentPen = [ -log2(1/9); -log2(1/9) ];

    for t = 3:length(valSeq)
        P = buildExperts(history,t,state,cfg,epsProb);
        p = P*w;
        p = safeNormalize(p,epsProb);

        T = temperatureFromRecentPenalties(recentPen,cfg.Wcal,cfg.Tmin,cfg.Tmax);
        p = p.^(1/T);
        p = safeNormalize(p,epsProb);

        trueSym = valSeq(t);
        pen = -log2(max(p(trueSym),epsProb));
        totalBits = totalBits + pen;

        pkTrue = max(P(trueSym,:)',epsProb);
        w = w .* exp(cfg.eta*log(pkTrue));
        w = w/sum(w);
        w = (1-cfg.lambda)*w + cfg.lambda*(ones(K,1)/K);
        w = w/sum(w);

        recentPen = [recentPen; pen]; %#ok<AGROW>
        if length(recentPen) > cfg.Wcal
            recentPen = recentPen(end-cfg.Wcal+1:end);
        end

        history = [history; trueSym]; %#ok<AGROW>
    end

    numSym = length(valSeq);
    bps = totalBits/numSym;
end

function state = trainAdaptiveExpertState(trainSeq,cfg)
    alphaLap = cfg.alphaLap;
    Ntr = length(trainSeq);

    % Global PMF
    globalCounts = alphaLap*ones(9,1);
    for i = 1:Ntr
        globalCounts(trainSeq(i)) = globalCounts(trainSeq(i)) + 1;
    end
    state.globalPMF = globalCounts/sum(globalCounts);

    % Run-length continuation
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
    state.pContinueByBucket = runCont./(runCont+runStop);

    % Periodicity counts
    periodList = 2:8;
    state.periodList = periodList;
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
    state.periodCounts = periodCounts;

    % Trend counts
    trendCounts = alphaLap*ones(5,9);
    for t = 3:Ntr
        d = trainSeq(t-1)-trainSeq(t-2);
        b = trendBin(d);
        trendCounts(b,trainSeq(t)) = trendCounts(b,trainSeq(t)) + 1;
    end
    state.trendCounts = trendCounts;

    % Volatility counts
    volCounts = alphaLap*ones(3,9);
    for t = max(3,cfg.Wtrend+1):Ntr
        recent = trainSeq(t-cfg.Wtrend:t-1);
        sdev = std(double(recent));
        v = volatilityState(sdev);
        volCounts(v,trainSeq(t)) = volCounts(v,trainSeq(t)) + 1;
    end
    state.volCounts = volCounts;

    % Nearest-pattern memory
    Lctx = cfg.Lctx;
    ctxMatrix = [];
    ctxNext = [];
    for t = Lctx+1:Ntr
        ctxMatrix = [ctxMatrix; trainSeq(t-Lctx:t-1).']; %#ok<AGROW>
        ctxNext = [ctxNext; trainSeq(t)]; %#ok<AGROW>
    end
    state.ctxMatrix = ctxMatrix;
    state.ctxNext = ctxNext;
end

function [bps,totalBits,N] = evaluateAdaptiveExpertWithSymbolMachine(state,testFile,cfg,verboseSM)
    epsProb = 1e-12;
    K = 9;

    N = initializeSymbolMachineS26(testFile,verboseSM);

    w = ones(K,1)/K;
    probs = ones(1,9)/9;
    [h1,~] = symbolMachineS26(probs);
    [h2,~] = symbolMachineS26(probs);
    history = [h1; h2];
    recentPen = [ -log2(1/9); -log2(1/9) ];

    for t = 3:N
        P = buildExperts(history,t,state,cfg,epsProb);
        p = P*w;
        p = safeNormalize(p,epsProb);

        T = temperatureFromRecentPenalties(recentPen,cfg.Wcal,cfg.Tmin,cfg.Tmax);
        p = p.^(1/T);
        p = safeNormalize(p,epsProb);

        [sym,pen] = symbolMachineS26(p.');

        pkTrue = max(P(sym,:)',epsProb);
        w = w .* exp(cfg.eta*log(pkTrue));
        w = w/sum(w);
        w = (1-cfg.lambda)*w + cfg.lambda*(ones(K,1)/K);
        w = w/sum(w);

        recentPen = [recentPen; pen]; %#ok<AGROW>
        if length(recentPen) > cfg.Wcal
            recentPen = recentPen(end-cfg.Wcal+1:end);
        end

        history = [history; sym]; %#ok<AGROW>
    end

    global SYMBOLDATA
    totalBits = SYMBOLDATA.totalPenaltyInBits;
    bps = totalBits/SYMBOLDATA.sequenceLength;

    reportSymbolMachineS26;
end

function P = buildExperts(history,t,state,cfg,epsProb)
    K = 9;
    P = zeros(9,K);

    P(:,1) = safeNormalize(state.globalPMF,epsProb);
    P(:,2) = expertShortWindow(history,cfg.Wshort,cfg.alphaLap,epsProb);
    P(:,3) = expertRunLength(history,state.pContinueByBucket,cfg.alphaLap,epsProb);
    P(:,4) = expertRecency(history,epsProb);
    P(:,5) = expertPeriodicity(t,state.periodList,state.periodCounts,epsProb);
    P(:,6) = expertTrend(history,state.trendCounts,epsProb);
    P(:,7) = expertVolatility(history,state.volCounts,cfg.Wtrend,epsProb);
    P(:,8) = expertNearestPattern(history,state.ctxMatrix,state.ctxNext,cfg.Lctx,cfg.alphaLap,cfg.Knear,epsProb);
    P(:,9) = ones(9,1)/9;
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
    if length(history) < 2
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

function p = expertNearestPattern(history,ctxMatrix,ctxNext,Lctx,alphaLap,Knear,epsProb)
    if length(history)<Lctx || isempty(ctxMatrix)
        p = ones(9,1)/9;
        return;
    end
    q = double(history(end-Lctx+1:end)).';
    D = sum(abs(double(ctxMatrix)-q),2);
    [Ds,idx] = sort(D,'ascend');
    Knear = min(Knear,length(idx));

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
        T = 1.0;
        return;
    end
    m = mean(recentPen(max(1,end-Wcal+1):end));
    ref = 2.5; % center reference
    delta = m - ref;
    T = 1 + 0.12*delta;
    T = max(Tmin,min(Tmax,T));
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
        v = 1;
    elseif sdev < 2.0
        v = 2;
    else
        v = 3;
    end
end