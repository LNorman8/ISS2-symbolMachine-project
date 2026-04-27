%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ADAPTIVE EXPERT MIXTURE PROJECT CHECKPOINT
%%% Non-MC / Non-AR forecasting framework for Symbol Machine S26.
%%% Colorado School of Mines EENG311 - Spring 2026 - Mike Wakin
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This script builds a forecasting model that is NOT explicitly a
% Markov Chain and NOT an Autoregressive model.
%
% Framework:
%   - Multiple "experts" each output a 9-symbol pmf forecast.
%   - An online meta-learner (exponential weights / Hedge) combines experts.
%   - Expert weights adapt over time based on log-loss performance.
%   - Optional probability calibration (temperature scaling) is included.
%
% Strengths:
%   - Adapts to nonstationary data
%   - Avoids MC sparse-context issues
%   - Avoids AR linear/Gaussian assumptions on symbols
%   - Always outputs valid nonzero probabilities (good for log-loss)
%
% NOTE:
%   - Training data can be loaded directly.
%   - Testing data must be accessed only through Symbol Machine calls.

clear all; clc;

%% USER SETTINGS

name = 'Hawaiian';
trainFile = ['sequence_' name '_train.mat'];
testFile  = ['sequence_' name '_test.mat'];

% Expert and learner hyperparameters
Wshort   = 25;     % short-window PMF window
Wtrend   = 12;     % window for trend/volatility features
Wcal     = 80;     % recent penalty window for temperature calibration
eta      = 0.45;   % learning rate for expert-weight update
lambda   = 0.01;   % weight-forgetting toward uniform
alphaLap = 0.5;    % Laplace smoothing pseudo-count
epsProb  = 1e-12;  % numerical floor

% Temperature calibration bounds
Tmin = 0.75;
Tmax = 1.35;

% Expert count (fixed by implementation below)
% 1: global PMF
% 2: short-window PMF
% 3: run-length continuation/switch expert
% 4: distance-to-last-occurrence (recency) expert
% 5: periodicity expert (periods 2..8)
% 6: trend-bin expert
% 7: volatility-state expert
% 8: nearest-pattern retrieval expert
% 9: uniform safety expert
K = 9;

%% TRAINING PHASE (build expert statistics from training sequence)

load(trainFile); % sequence is available because this is TRAINING data
trainSeq = sequence(:);
Ntrain = length(trainSeq);

if any(trainSeq < 1) || any(trainSeq > 9) || any(abs(trainSeq-round(trainSeq))>0)
    error('Training sequence must contain integer symbols 1..9.');
end

% --- Global PMF expert stats ---
globalCounts = alphaLap*ones(9,1);
for i = 1:Ntrain
    globalCounts(trainSeq(i)) = globalCounts(trainSeq(i)) + 1;
end
globalPMF = globalCounts / sum(globalCounts);

% --- Run-length expert stats ---
% continuation prob by run length bucket
% buckets: 1,2,3,4,>=5
numRunBuckets = 5;
runCont = ones(numRunBuckets,1); % continuation counts
runStop = ones(numRunBuckets,1); % stop counts

r = 1;
while r <= Ntrain
    s = trainSeq(r);
    j = r+1;
    while j<=Ntrain && trainSeq(j)==s
        j = j+1;
    end
    runLen = j-r;  % run of symbol s with length runLen
    b = min(runLen,5);
    % For each position within run except final one -> continuation event
    if runLen >= 2
        runCont(b) = runCont(b) + (runLen-1);
    end
    % Final position transitions to new symbol if exists
    if j<=Ntrain
        runStop(b) = runStop(b) + 1;
    end
    r = j;
end
pContinueByBucket = runCont ./ (runCont + runStop);

% --- Periodicity expert stats: symbol counts conditioned on t mod p ---
periodList = 2:8;
numPeriods = length(periodList);
periodCounts = cell(numPeriods,1);
for ip = 1:numPeriods
    p = periodList(ip);
    % rows = phase 1..p, cols = symbol 1..9
    C = alphaLap*ones(p,9);
    for t = 1:Ntrain
        ph = mod(t-1,p)+1;
        C(ph,trainSeq(t)) = C(ph,trainSeq(t)) + 1;
    end
    periodCounts{ip} = C;
end

% --- Trend-bin expert stats ---
% trend bin based on delta = x(t)-x(t-1):
% bin1: delta<=-2, bin2: -1, bin3:0, bin4:+1, bin5:>=+2
numTrendBins = 5;
trendCounts = alphaLap*ones(numTrendBins,9);
for t = 3:Ntrain
    d = trainSeq(t-1)-trainSeq(t-2);
    b = trendBin(d);
    trendCounts(b,trainSeq(t)) = trendCounts(b,trainSeq(t)) + 1;
end

% --- Volatility-state expert stats ---
% state based on std of last Wtrend symbols (at t-1):
% low, medium, high volatility via fixed thresholds in symbol-space
volCounts = alphaLap*ones(3,9);
for t = max(3,Wtrend+1):Ntrain
    recent = trainSeq(t-Wtrend:t-1);
    sdev = std(double(recent));
    vState = volatilityState(sdev);
    volCounts(vState,trainSeq(t)) = volCounts(vState,trainSeq(t)) + 1;
end

% --- Nearest-pattern retrieval memory ---
% store training contexts of length Lctx and the following symbol
Lctx = 6;
ctxMatrix = [];
ctxNext   = [];
for t = Lctx+1:Ntrain
    ctx = trainSeq(t-Lctx:t-1).';
    ctxMatrix = [ctxMatrix; ctx];
    ctxNext   = [ctxNext; trainSeq(t)];
end

fprintf('Training complete on %s with %d symbols.\n',trainFile,Ntrain);

%% TESTING PHASE WITH SYMBOL MACHINE

sequenceLength = initializeSymbolMachineS26(testFile);

% Initialize expert weights uniformly
w = ones(K,1)/K;

% Initialize rolling history from first two symbols (uniform probs)
probs = ones(1,9)/9;
[hist1,~] = symbolMachineS26(probs);
[hist2,~] = symbolMachineS26(probs);
history = [hist1; hist2];  % observed test symbols so far (column)

% For calibration tracking
recentPenalties = [];

for t = 3:sequenceLength

    % Build all expert distributions as columns of P (9 x K)
    P = zeros(9,K);

    % Expert 1: global PMF (fixed from training)
    P(:,1) = safeNormalize(globalPMF,epsProb);

    % Expert 2: short-window PMF over recent observed history
    P(:,2) = expertShortWindow(history,Wshort,alphaLap,epsProb);

    % Expert 3: run-length continuation/switch
    P(:,3) = expertRunLength(history,pContinueByBucket,alphaLap,epsProb);

    % Expert 4: distance-to-last-occurrence / recency
    P(:,4) = expertRecency(history,epsProb);

    % Expert 5: periodicity (mixture across periods)
    P(:,5) = expertPeriodicity(t,periodList,periodCounts,epsProb);

    % Expert 6: trend-bin
    P(:,6) = expertTrend(history,trendCounts,alphaLap,epsProb);

    % Expert 7: volatility-state
    P(:,7) = expertVolatility(history,volCounts,Wtrend,alphaLap,epsProb);

    % Expert 8: nearest-pattern retrieval
    P(:,8) = expertNearestPattern(history,ctxMatrix,ctxNext,Lctx,alphaLap,epsProb);

    % Expert 9: uniform safety
    P(:,9) = ones(9,1)/9;

    % Mixture forecast
    probsCol = P*w;
    probsCol = safeNormalize(probsCol,epsProb);

    % Optional online temperature calibration
    T = temperatureFromRecentPenalties(recentPenalties,Wcal,Tmin,Tmax);
    probsCol = probsCol.^(1/T);
    probsCol = safeNormalize(probsCol,epsProb);

    probs = probsCol.'; % Symbol machine accepts row or column
    [symbol,penalty] = symbolMachineS26(probs);

    % Update expert weights using observed symbol
    pkTrue = max(P(symbol,:)',epsProb);
    w = w .* exp(eta*log(pkTrue));
    w = w / sum(w);

    % Forgetting toward uniform for nonstationarity robustness
    w = (1-lambda)*w + lambda*(ones(K,1)/K);
    w = w / sum(w);

    % Update calibration history
    recentPenalties = [recentPenalties; penalty];
    if length(recentPenalties) > Wcal
        recentPenalties = recentPenalties(end-Wcal+1:end);
    end

    % Append observed symbol to history
    history = [history; symbol];
end

reportSymbolMachineS26;

%% OPTIONAL: DISPLAY FINAL EXPERT WEIGHTS
disp('Final expert weights (1..9):');
disp(w.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LOCAL FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

    % If continue: place mass on last symbol.
    % If switch: spread mass across others via weak frequency prior.
    counts = alphaLap*ones(9,1);
    switchPMF = counts / sum(counts);
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
            scores(s) = 1;  % unseen gets base score
        else
            gap = n - idx + 1;
            scores(s) = 1 / gap; % more recent => larger score
        end
    end
    p = safeNormalize(scores,epsProb);
end

function p = expertPeriodicity(t,periodList,periodCounts,epsProb)
    numPeriods = length(periodList);
    mix = zeros(9,1);
    wPer = ones(numPeriods,1)/numPeriods; % equal period weights
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

function p = expertTrend(history,trendCounts,alphaLap,epsProb)
    if length(history) < 2
        p = ones(9,1)/9;
        return;
    end
    d = history(end)-history(end-1);
    b = trendBin(d);
    row = trendCounts(b,:).';
    row = row + 0*alphaLap; %#ok<NASGU> % keeps style consistent
    p = safeNormalize(row,epsProb);
end

function p = expertVolatility(history,volCounts,Wtrend,alphaLap,epsProb)
    n = length(history);
    if n < max(3,Wtrend)
        p = ones(9,1)/9;
        return;
    end
    recent = history(max(1,n-Wtrend+1):n);
    sdev = std(double(recent));
    vState = volatilityState(sdev);
    row = volCounts(vState,:).';
    row = row + 0*alphaLap; %#ok<NASGU>
    p = safeNormalize(row,epsProb);
end

function p = expertNearestPattern(history,ctxMatrix,ctxNext,Lctx,alphaLap,epsProb)
    if length(history) < Lctx || isempty(ctxMatrix)
        p = ones(9,1)/9;
        return;
    end
    q = double(history(end-Lctx+1:end)).'; % 1 x Lctx
    D = sum(abs(double(ctxMatrix) - q),2); % L1 distance
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
    % thresholds in symbol space (1..9)
    if sdev < 1.0
        v = 1; % low
    elseif sdev < 2.0
        v = 2; % medium
    else
        v = 3; % high
    end
end

function p = safeNormalize(x,epsProb)
    x = x(:);
    x = max(x,epsProb);
    p = x / sum(x);
end

function T = temperatureFromRecentPenalties(recentPenalties,Wcal,Tmin,Tmax)
    % Simple adaptive calibration:
    % - If penalties recently high, slightly flatten (T>1).
    % - If penalties recently low, slightly sharpen (T<1).
    if isempty(recentPenalties)
        T = 1.0;
        return;
    end
    m = mean(recentPenalties(max(1,end-Wcal+1):end));

    % Reference around ~2.5 bits/symbol for rough centering in 9-symbol tasks
    % (uniform is 3.17 bits/symbol)
    ref = 2.5;
    delta = m - ref;

    % map delta to temperature gently
    T = 1 + 0.12*delta;
    T = max(Tmin,min(Tmax,T));
end