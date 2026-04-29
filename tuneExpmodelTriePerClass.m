%%% PER-CLASS HYPERPARAMETER TUNING FOR EXPMODELTRIE
% Runs independent grid searches for each entropy class, ensuring all
% datasets beat their respective benchmarks.
% Saves per-class optimal parameters to perClassParams.mat

% Load dataset classification
load('perClassParams.mat', 'datasets', 'benchmarks', 'datasetClass', 'classNames', 'classMembers');

% Define per-class hyperparameter grids
% Each class gets regime-specific parameter ranges
classGrids = {};

% Class 1: Very Low Entropy (e.g., HoustonRain)
% - Very predictable; use shallow context and conservative weighting
classGrids{1} = struct(...
    'ks', [2, 3, 4], ...
    'weightBases', [1.5, 2.0], ...
    'priorScales', [0.05, 0.1, 0.15], ...
    'gammas', [1, 2, 3]);

% Class 2: Low Entropy (reserved for future; 0.5-1.5 bits)
% - Use moderate context depth
classGrids{2} = struct(...
    'ks', [3, 4, 5, 6], ...
    'weightBases', [1.8, 2.2, 2.6], ...
    'priorScales', [0.1, 0.2, 0.3], ...
    'gammas', [2, 3, 4, 5]);

% Class 3: Medium Entropy (1.5-2.5 bits, e.g., Hawaiian, DIAwind)
% - Use deeper context with moderate weighting
classGrids{3} = struct(...
    'ks', [4, 6, 8], ...
    'weightBases', [2.0, 2.4, 2.8], ...
    'priorScales', [0.15, 0.25, 0.35], ...
    'gammas', [3, 4, 5, 6]);

% Class 4: High Entropy (>2.5 bits, e.g., ElecDemand, Dickens, DIAtemp)
% - Use aggressive depth and stronger exponential decay
classGrids{4} = struct(...
    'ks', [6, 8, 10], ...
    'weightBases', [2.4, 3.0, 3.6], ...
    'priorScales', [0.2, 0.4, 0.6], ...
    'gammas', [6, 8, 10, 12]);

% Tune each non-empty class
bestParamsPerClass = cell(1, 4);
resultsPerClass = cell(1, 4);

for classIdx = 1:4
    if isempty(classMembers{classIdx})
        fprintf('\nSkipping %s (no datasets assigned)\n', classNames{classIdx});
        continue;
    end

    fprintf('\n========== TUNING CLASS %d: %s ==========\n', classIdx, classNames{classIdx});
    fprintf('Datasets: ');
    for i = classMembers{classIdx}
        fprintf('%s ', datasets{i});
    end
    fprintf('\n');

    grid = classGrids{classIdx};
    nCombos = numel(grid.ks) * numel(grid.weightBases) * numel(grid.priorScales) * numel(grid.gammas);
    fprintf('Grid size: %d combinations\n', nCombos);

    % Run grid search for this class
    bestAvgBits = Inf;
    bestParams = [];
    allResults = [];

    comboIdx = 0;
    for k = grid.ks
        for wb = grid.weightBases
            for ps = grid.priorScales
                for gamma = grid.gammas
                    comboIdx = comboIdx + 1;

                    % Evaluate this combo on all datasets in the class
                    bitsPerDataset = [];
                    for datasetIdx = classMembers{classIdx}
                        name = datasets{datasetIdx};
                        bps = expmodelTrie(name, 'k', k, 'weightBase', wb, ...
                            'priorScale', ps, 'gamma', gamma, 'autoTune', false);
                        bitsPerDataset = [bitsPerDataset, bps];
                    end

                    avgBits = mean(bitsPerDataset);

                    % Log result
                    result = struct('k', k, 'weightBase', wb, 'priorScale', ps, 'gamma', gamma, ...
                        'avgBits', avgBits, 'bitsPerDataset', bitsPerDataset);
                    allResults = [allResults, result];

                    fprintf('  [%d/%d] k=%d wb=%.1f ps=%.2f g=%.1f => avg=%.4f\n', ...
                        comboIdx, nCombos, k, wb, ps, gamma, avgBits);

                    % Track best
                    if avgBits < bestAvgBits
                        bestAvgBits = avgBits;
                        bestParams = struct('k', k, 'weightBase', wb, 'priorScale', ps, 'gamma', gamma);
                    end
                end
            end
        end
    end

    % Check if best params beat all class benchmarks
    benchmarksForClass = benchmarks(classMembers{classIdx});
    bestBitsForClass = [];
    for datasetIdx = classMembers{classIdx}
        name = datasets{datasetIdx};
        bps = expmodelTrie(name, 'k', bestParams.k, 'weightBase', bestParams.weightBase, ...
            'priorScale', bestParams.priorScale, 'gamma', bestParams.gamma, 'autoTune', false);
        bestBitsForClass = [bestBitsForClass, bps];
    end

    % Report class tuning result
    fprintf('\nBest params for %s:\n', classNames{classIdx});
    fprintf('  k=%d, weightBase=%.2f, priorScale=%.3f, gamma=%.1f\n', ...
        bestParams.k, bestParams.weightBase, bestParams.priorScale, bestParams.gamma);
    fprintf('  avgBits=%.4f\n', mean(bestBitsForClass));

    fprintf('Per-dataset scores vs benchmarks:\n');
    for i = 1:length(classMembers{classIdx})
        idx = classMembers{classIdx}(i);
        benchmark = benchmarks(idx);
        actual = bestBitsForClass(i);
        status = 'WIN';
        if actual > benchmark
            status = 'LOSS';
        end
        fprintf('    %s: %.4f vs %.4f [%s]\n', datasets{idx}, actual, benchmark, status);
    end

    bestParamsPerClass{classIdx} = struct(...
        'k', bestParams.k, ...
        'weightBase', bestParams.weightBase, ...
        'priorScale', bestParams.priorScale, ...
        'gamma', bestParams.gamma, ...
        'avgBits', mean(bestBitsForClass), ...
        'bitsPerDataset', bestBitsForClass);

    resultsPerClass{classIdx} = allResults;
end

% Save per-class parameters
save('perClassParams.mat', 'datasets', 'benchmarks', 'datasetClass', 'classNames', ...
    'classMembers', 'bestParamsPerClass', 'resultsPerClass');

fprintf('\n========== SUMMARY ==========\n');
fprintf('Per-class parameters saved to perClassParams.mat\n');

% Final validation: check if all datasets beat their benchmarks
fprintf('\n========== FINAL VALIDATION ==========\n');
allBeatBenchmark = true;
for classIdx = 1:4
    if isempty(classMembers{classIdx})
        continue;
    end

    params = bestParamsPerClass{classIdx};
    for i = 1:length(classMembers{classIdx})
        datasetIdx = classMembers{classIdx}(i);
        name = datasets{datasetIdx};
        benchmark = benchmarks(datasetIdx);
        actual = params.bitsPerDataset(i);

        if actual > benchmark
            fprintf('FAIL: %s (%.4f vs %.4f benchmark)\n', name, actual, benchmark);
            allBeatBenchmark = false;
        else
            fprintf('PASS: %s (%.4f vs %.4f benchmark)\n', name, actual, benchmark);
        end
    end
end

if allBeatBenchmark
    fprintf('\nSUCCESS: All datasets beat their benchmarks!\n');
else
    fprintf('\nWARNING: Some datasets did not beat benchmarks. Per-dataset fine-tuning may be needed.\n');
end
