%% FINAL BENCHMARK TEST - All 7 datasets with per-class tuning
% Tests all datasets using the trained per-class hyperparameters

fprintf('\n========== FINAL BENCHMARK TEST (Per-Class Tuning) ==========\n\n');

datasets = {'Hawaiian', 'ElecDemand', 'Dickens', 'DIAtemp', 'DIAwind', 'solarWind', 'HoustonRain'};
benchmarks = [1.6456, 1.3387, 1.7674, 1.6797, 1.8762, 0.6692, 0.1012];

% Load per-class parameters from tuning
data = load('perClassParams.mat');

results_table = [];
wins = 0;
total_gap = 0;

for i = 1:length(datasets)
    name = datasets{i};
    benchmark = benchmarks(i);

    fprintf('Testing %s...\n', name);

    % Run with usePerClassTuning=true to use trained parameters
    bps = expmodelTrie(name, 'usePerClassTuning', 'true');

    gap = benchmark - bps;
    status = 'LOSS';
    if bps <= benchmark
        status = 'WIN';
        wins = wins + 1;
        total_gap = total_gap + gap;
    end

    results_table = [results_table; struct('name', name, 'bps', bps, ...
        'benchmark', benchmark, 'gap', gap, 'status', status)];

    fprintf('  %s: %.4f vs %.4f [%s] (%+.4f)\n\n', name, bps, benchmark, status, gap);
end

fprintf('\n========== SUMMARY ==========\n');
fprintf('%-15s | BPS      | Benchmark | Gap      | Status\n', 'Dataset');
fprintf('%-15s-+----------+-----------+----------+------\n', repmat('-', 1, 15));

for i = 1:length(datasets)
    r = results_table(i);
    fprintf('%-15s | %8.4f | %9.4f | %+8.4f | %s\n', ...
        r.name, r.bps, r.benchmark, r.gap, r.status);
end

fprintf('\n✓ RESULTS: %d/7 benchmarks beat\n', wins);
if wins > 0
    fprintf('  Average gap on winners: %.4f bits\n', total_gap/wins);
end
