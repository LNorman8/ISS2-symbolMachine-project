function datasetParams = tuneExpmodelTriePerDatasetTrainOnly()
% Per-dataset training-only tuning with dataset-name lookup support.

load('perClassParams.mat', 'datasets', 'benchmarks', 'entropies', 'datasetClass', ...
    'classNames', 'classThresholds');

mixAlphas = [0.00, 0.15, 0.25, 0.35];
classGrids = cell(1, 4);
classGrids{1} = struct('ks', [2, 3, 4], 'weightBases', [1.5, 2.0], 'priorScales', [0.05, 0.10, 0.15], 'gammas', [1, 2, 3]);
classGrids{2} = struct('ks', [3, 4, 5, 6], 'weightBases', [1.8, 2.2, 2.6], 'priorScales', [0.10, 0.20, 0.30], 'gammas', [2, 3, 4, 5]);
classGrids{3} = struct('ks', [4, 6, 8], 'weightBases', [2.0, 2.4, 2.8], 'priorScales', [0.15, 0.25, 0.35], 'gammas', [3, 4, 5, 6]);
classGrids{4} = struct('ks', [6, 8, 10], 'weightBases', [2.4, 3.0, 3.6], 'priorScales', [0.20, 0.40, 0.60], 'gammas', [6, 8, 10, 12]);

datasetParams = struct('name', {}, 'entropy', {}, 'classIdx', {}, 'benchmark', {}, 'params', {}, 'cvBits', {}, 'testBits', {});

fprintf('\n========== DATASET-SPECIFIC TUNING (TRAINING DATA ONLY) ==========');

for di = 1:numel(datasets)
    name = datasets{di};
    classIdx = datasetClass(di);
    grid = classGrids{classIdx};
    nCombos = numel(grid.ks) * numel(grid.weightBases) * numel(grid.priorScales) * numel(grid.gammas) * numel(mixAlphas);

    fprintf('\n\n========== TUNING DATASET: %s (class=%s) ==========', name, classNames{classIdx});
    fprintf('\nGrid size: %d combinations\n', nCombos);

    bestScore = Inf;
    bestParams = struct('k', [], 'weightBase', [], 'priorScale', [], 'gamma', [], 'mixAlpha', [], 'shallowK', []);
    comboIdx = 0;

    for k = grid.ks
        for wb = grid.weightBases
            for ps = grid.priorScales
                for gamma = grid.gammas
                    for mixAlpha = mixAlphas
                        comboIdx = comboIdx + 1;
                        cvBits = expmodelTrieTrainCV(name, k, wb, ps, gamma, 0.2, mixAlpha);
                        fprintf('  [%d/%d] k=%d wb=%.1f ps=%.2f g=%.1f mix=%.2f => cv=%.4f\n', ...
                            comboIdx, nCombos, k, wb, ps, gamma, mixAlpha, cvBits);
                        if cvBits < bestScore
                            bestScore = cvBits;
                            bestParams.k = k;
                            bestParams.weightBase = wb;
                            bestParams.priorScale = ps;
                            bestParams.gamma = gamma;
                            bestParams.mixAlpha = mixAlpha;
                            bestParams.shallowK = defaultShallowK(k);
                        end
                    end
                end
            end
        end
    end

    [testBits, selectedParams] = expmodelTrie(name, ...
        'autoTune', false, ...
        'useDatasetTuning', false, ...
        'usePerClassTuning', false, ...
        'useGlobalDefault', false, ...
        'k', bestParams.k, ...
        'weightBase', bestParams.weightBase, ...
        'priorScale', bestParams.priorScale, ...
        'gamma', bestParams.gamma, ...
        'mixAlpha', bestParams.mixAlpha, ...
        'shallowK', bestParams.shallowK);

    datasetParams(di).name = name;
    datasetParams(di).entropy = entropies(di);
    datasetParams(di).classIdx = classIdx;
    datasetParams(di).benchmark = benchmarks(di);
    datasetParams(di).params = selectedParams;
    datasetParams(di).cvBits = bestScore;
    datasetParams(di).testBits = testBits;

    fprintf('Best params for %s: k=%d wb=%.2f ps=%.3f g=%.1f mix=%.2f\n', ...
        name, bestParams.k, bestParams.weightBase, bestParams.priorScale, bestParams.gamma, bestParams.mixAlpha);
    fprintf('  CV bits=%.4f  Test bits=%.4f  Benchmark=%.4f\n', bestScore, testBits, benchmarks(di));
end

save('datasetParams.mat', 'datasetParams', 'datasets', 'benchmarks', 'entropies', 'datasetClass', 'classNames', 'classThresholds');

fprintf('\nDataset-specific tuning saved to datasetParams.mat\n');
end

function shallowK = defaultShallowK(k)
shallowK = min(2, max(1, k - 1));
end
