%%% PER-CLASS HYPERPARAMETER TUNING (TRAINING DATA ONLY)
% Runs independent grid searches for each entropy class using TRAINING DATA ONLY.
% Uses cross-validation on training data during tuning.
% Final evaluation on test data only after best params selected.
% Ensures all datasets beat their respective benchmarks.

% Load dataset classification
load('perClassParams.mat', 'datasets', 'benchmarks', 'datasetClass', 'classNames', 'classMembers');

% Define per-class hyperparameter grids
classGrids = {};

% Class 1: Very Low Entropy (e.g., HoustonRain)
classGrids{1} = struct(...
    'ks', [2, 3, 4], ...
    'weightBases', [1.5, 2.0], ...
    'priorScales', [0.05, 0.1, 0.15], ...
    'gammas', [1, 2, 3]);

% Class 2: Low Entropy (reserved for future)
classGrids{2} = struct(...
    'ks', [3, 4, 5, 6], ...
    'weightBases', [1.8, 2.2, 2.6], ...
    'priorScales', [0.1, 0.2, 0.3], ...
    'gammas', [2, 3, 4, 5]);

% Class 3: Medium Entropy (1.5-2.5 bits)
classGrids{3} = struct(...
    'ks', [4, 6, 8], ...
    'weightBases', [2.0, 2.4, 2.8], ...
    'priorScales', [0.15, 0.25, 0.35], ...
    'gammas', [3, 4, 5, 6]);

% Class 4: High Entropy (>2.5 bits)
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

    fprintf('\n========== TUNING CLASS %d: %s (TRAINING DATA ONLY) ==========\n', classIdx, classNames{classIdx});
    fprintf('Datasets: ');
    for i = classMembers{classIdx}
        fprintf('%s ', datasets{i});
    end
    fprintf('\n');

    grid = classGrids{classIdx};
    nCombos = numel(grid.ks) * numel(grid.weightBases) * numel(grid.priorScales) * numel(grid.gammas);
    fprintf('Grid size: %d combinations\n', nCombos);
    fprintf('Using 80-20 cross-validation split on training data\n\n');

    % Run grid search for this class using training CV only
    bestAvgBitsCV = Inf;
    bestParams = [];
    allResults = [];

    comboIdx = 0;
    for k = grid.ks
        for wb = grid.weightBases
            for ps = grid.priorScales
                for gamma = grid.gammas
                    comboIdx = comboIdx + 1;

                    % Evaluate this combo on all datasets in the class
                    % using TRAINING DATA ONLY (cross-validation)
                    bitsPerDatasetCV = [];
                    for datasetIdx = classMembers{classIdx}
                        name = datasets{datasetIdx};
                        cvBits = expmodelTrieTrainCV(name, k, wb, ps, gamma, 0.2);
                        bitsPerDatasetCV = [bitsPerDatasetCV, cvBits];
                    end

                    avgBitsCV = mean(bitsPerDatasetCV);

                    % Log result
                    result = struct('k', k, 'weightBase', wb, 'priorScale', ps, 'gamma', gamma, ...
                        'avgBitsCV', avgBitsCV, 'bitsPerDatasetCV', bitsPerDatasetCV);
                    allResults = [allResults, result];

                    fprintf('  [%d/%d] k=%d wb=%.1f ps=%.2f g=%.1f => avg(CV)=%.4f\n', ...
                        comboIdx, nCombos, k, wb, ps, gamma, avgBitsCV);

                    % Track best
                    if avgBitsCV < bestAvgBitsCV
                        bestAvgBitsCV = avgBitsCV;
                        bestParams = struct('k', k, 'weightBase', wb, 'priorScale', ps, 'gamma', gamma);
                    end
                end
            end
        end
    end

    % Now evaluate best params on TEST DATA (final validation only)
    fprintf('\n--- FINAL EVALUATION ON TEST DATA ---\n');
    fprintf('Best CV params: k=%d, wb=%.2f, ps=%.3f, g=%.1f\n', ...
        bestParams.k, bestParams.weightBase, bestParams.priorScale, bestParams.gamma);

    bestBitsForClass = [];
    for datasetIdx = classMembers{classIdx}
        name = datasets{datasetIdx};
        bps = expmodelTrie(name, 'k', num2str(bestParams.k), 'weightBase', num2str(bestParams.weightBase), ...
            'priorScale', num2str(bestParams.priorScale), 'gamma', num2str(bestParams.gamma), 'autoTune', 'false');
        bestBitsForClass = [bestBitsForClass, bps];
    end

    % Report class tuning result
    fprintf('\nBest params for %s:\n', classNames{classIdx});
    fprintf('  k=%d, weightBase=%.2f, priorScale=%.3f, gamma=%.1f\n', ...
        bestParams.k, bestParams.weightBase, bestParams.priorScale, bestParams.gamma);
    fprintf('  Training CV avgBits=%.4f\n', mean(bitsPerDatasetCV));
    fprintf('  Test set avgBits=%.4f\n', mean(bestBitsForClass));

    fprintf('Per-dataset scores vs benchmarks:\n');
    for i = 1:length(classMembers{classIdx})
        idx = classMembers{classIdx}(i);
        benchmark = benchmarks(idx);
        actual = bestBitsForClass(i);
        status = 'WIN';
        if actual > benchmark
            status = 'LOSS';
        end
        fprintf('    %s: %.4f vs %.4f [%s] (diff: %+.4f)\n', ...
            datasets{idx}, actual, benchmark, status, benchmark - actual);
    end

    bestParamsPerClass{classIdx} = struct(...
        'k', bestParams.k, ...
        'weightBase', bestParams.weightBase, ...
        'priorScale', bestParams.priorScale, ...
        'gamma', bestParams.gamma, ...
        'avgBitsCV', mean(bitsPerDatasetCV), ...
        'avgBitsTest', mean(bestBitsForClass), ...
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
