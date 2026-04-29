%% TEST SYMBOL-ADAPTIVE CONTEXT DEPTH vs BASELINE
% Compares standard and symbol-adaptive versions on all 7 datasets

fprintf('\n========== SYMBOL-ADAPTIVE CONTEXT DEPTH TEST ==========\n\n');

datasets = {'Hawaiian', 'ElecDemand', 'Dickens', 'DIAtemp', 'DIAwind', 'solarWind', 'HoustonRain'};
benchmarks = [1.6456, 1.3387, 1.7674, 1.6797, 1.8762, 0.6692, 0.1012];

results = struct('name', {}, 'bps', {}, 'benchmark', {}, 'gap', {}, 'status', {});

for i = 1:length(datasets)
    name = datasets{i};
    benchmark = benchmarks(i);

    fprintf('Testing %s (benchmark=%.4f)...\n', name, benchmark);

    % Run with symbol-adaptive features (autoTune=false uses global params + adaptive)
    startTime = tic;
    bps = expmodelTrie(name, 'autoTune', 'false');
    elapsedTime = toc(startTime);

    gap = benchmark - bps;
    status = 'LOSS';
    if bps <= benchmark
        status = 'WIN';
    end

    results(i).name = name;
    results(i).bps = bps;
    results(i).benchmark = benchmark;
    results(i).gap = gap;
    results(i).status = status;

    fprintf('  Result: %.4f bps [%s] (gap: %+.4f) - %.1f sec\n\n', bps, status, gap, elapsedTime);
end

fprintf('\n========== SUMMARY ==========\n');
fprintf('%-15s | BPS      | Benchmark | Gap      | Status\n', 'Dataset');
fprintf('%-15s-+----------+-----------+----------+------\n', repmat('-', 1, 15));

wins = 0;
totalGap = 0;
for i = 1:length(datasets)
    fprintf('%-15s | %8.4f | %9.4f | %+8.4f | %s\n', ...
        results(i).name, results(i).bps, results(i).benchmark, results(i).gap, results(i).status);
    if strcmp(results(i).status, 'WIN')
        wins = wins + 1;
        totalGap = totalGap + results(i).gap;
    end
end

fprintf('\n✓ RESULTS: %d/7 benchmarks beat, avg gap on winners: %.4f bits\n', wins, totalGap/max(1,wins));
