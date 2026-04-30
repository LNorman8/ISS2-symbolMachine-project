function datasetParams = tuneExpmodelTriePerDatasetNoMixTrainOnly(useParallel)
% Per-dataset training-only tuning baseline with depth mixture disabled.
% Keeps the dataset-specific split logic, but fixes mixAlpha at 0 so the
% resulting baseline measures the core trie model without depth blending.

if nargin < 1 || isempty(useParallel)
    useParallel = canUseParallelTuning();
end

load('perClassParams.mat', 'datasets', 'benchmarks', 'entropies', 'datasetClass', ...
    'classNames', 'classThresholds');

classGrids = cell(1, 4);
classGrids{1} = struct('ks', [2, 3, 4], 'weightBases', [1.5, 2.0], 'priorScales', [0.05, 0.10, 0.15], 'gammas', [1, 2, 3]);
classGrids{2} = struct('ks', [3, 4, 5, 6], 'weightBases', [1.8, 2.2, 2.6], 'priorScales', [0.10, 0.20, 0.30], 'gammas', [2, 3, 4, 5]);
classGrids{3} = struct('ks', [4, 6, 8], 'weightBases', [2.0, 2.4, 2.8], 'priorScales', [0.15, 0.25, 0.35], 'gammas', [3, 4, 5, 6]);
classGrids{4} = struct('ks', [6, 8, 10], 'weightBases', [2.4, 3.0, 3.6], 'priorScales', [0.20, 0.40, 0.60], 'gammas', [6, 8, 10, 12]);

fprintf('\n========== DATASET-SPECIFIC TUNING BASELINE (NO DEPTH MIXTURE) ==========');
if useParallel
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool('local');
    end
    fprintf('\nParallel tuning enabled with %d workers\n', pool.NumWorkers);
else
    fprintf('\nParallel tuning disabled; running serially\n');
end

datasetResults = cell(1, numel(datasets));

if useParallel
    parfor di = 1:numel(datasets)
        datasetResults{di} = tuneOneDatasetNoMix( ...
            datasets{di}, datasetClass(di), benchmarks(di), entropies(di), ...
            classNames{datasetClass(di)}, classGrids{datasetClass(di)}, false);
    end
else
    for di = 1:numel(datasets)
        datasetResults{di} = tuneOneDatasetNoMix( ...
            datasets{di}, datasetClass(di), benchmarks(di), entropies(di), ...
            classNames{datasetClass(di)}, classGrids{datasetClass(di)}, true);
    end
end

datasetParams = [datasetResults{:}];

save('datasetParamsNoMix.mat', 'datasetParams', 'datasets', 'benchmarks', 'entropies', 'datasetClass', 'classNames', 'classThresholds');
save('datasetParams.mat', 'datasetParams', 'datasets', 'benchmarks', 'entropies', 'datasetClass', 'classNames', 'classThresholds');

fprintf('\nDataset-specific no-mix tuning saved to datasetParamsNoMix.mat and datasetParams.mat\n');
try
    updateResultsMarkdown;
catch err
    fprintf('WARNING: results.md was not refreshed automatically: %s\n', err.message);
end
end

function result = tuneOneDatasetNoMix(name, classIdx, benchmark, entropy, className, grid, verbose)
nCombos = numel(grid.ks) * numel(grid.weightBases) * numel(grid.priorScales) * numel(grid.gammas);
[kGrid, wbGrid, psGrid, gammaGrid] = ndgrid(grid.ks, grid.weightBases, grid.priorScales, grid.gammas);
comboK = kGrid(:);
comboWB = wbGrid(:);
comboPS = psGrid(:);
comboGamma = gammaGrid(:);

if verbose
    fprintf('\n\n========== TUNING DATASET: %s (class=%s) ==========', name, className);
    fprintf('\nGrid size: %d combinations\n', nCombos);
    fprintf('Depth mixture is disabled for this baseline (mixAlpha = 0.0)\n');
else
    fprintf('Tuning %s (%s): %d combinations\n', name, className, nCombos);
end

bestScore = Inf;
bestIdx = 1;
scores = NaN(nCombos, 1);

for comboIdx = 1:nCombos
    scores(comboIdx) = expmodelTrieTrainCV(name, comboK(comboIdx), comboWB(comboIdx), ...
        comboPS(comboIdx), comboGamma(comboIdx), 0.2, 0.0, 1);
    if verbose
        fprintf('  [%d/%d] k=%d wb=%.1f ps=%.2f g=%.1f => cv=%.4f\n', ...
            comboIdx, nCombos, comboK(comboIdx), comboWB(comboIdx), comboPS(comboIdx), comboGamma(comboIdx), scores(comboIdx));
    end
    if scores(comboIdx) < bestScore
        bestScore = scores(comboIdx);
        bestIdx = comboIdx;
    end
end

bestParams = struct('k', comboK(bestIdx), 'weightBase', comboWB(bestIdx), 'priorScale', comboPS(bestIdx), ...
    'gamma', comboGamma(bestIdx), 'mixAlpha', 0.0, 'shallowK', 1);

[testBits, selectedParams] = expmodelTrie(name, ...
    'autoTune', false, ...
    'useDatasetTuning', false, ...
    'usePerClassTuning', false, ...
    'useGlobalDefault', false, ...
    'k', bestParams.k, ...
    'weightBase', bestParams.weightBase, ...
    'priorScale', bestParams.priorScale, ...
    'gamma', bestParams.gamma, ...
    'mixAlpha', 0.0, ...
    'shallowK', 1);

result = struct();
result.name = name;
result.entropy = entropy;
result.classIdx = classIdx;
result.benchmark = benchmark;
result.params = selectedParams;
result.cvBits = bestScore;
result.testBits = testBits;

if verbose
    fprintf('Best params for %s: k=%d wb=%.2f ps=%.3f g=%.1f mix=0.00\n', ...
        name, bestParams.k, bestParams.weightBase, bestParams.priorScale, bestParams.gamma);
    fprintf('  CV bits=%.4f  Test bits=%.4f  Benchmark=%.4f\n', bestScore, testBits, benchmark);
end
end

function tf = canUseParallelTuning()
tf = license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('parallel'));
end