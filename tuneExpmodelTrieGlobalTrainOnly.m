function globalDefaultParams = tuneExpmodelTrieGlobalTrainOnly()
% Training-only global default tuning across all datasets.

load('perClassParams.mat', 'datasets', 'benchmarks');

ks = [5, 6, 7];
weightBases = [2.0, 2.2, 2.4];
priorScales = [0.10, 0.15, 0.20, 0.25];
gammas = [2.0, 3.0, 4.0, 5.0];
mixAlphas = [0.00, 0.15, 0.25, 0.35];

totalCombos = numel(ks) * numel(weightBases) * numel(priorScales) * numel(gammas) * numel(mixAlphas);
comboIdx = 0;
results = struct('k', {}, 'weightBase', {}, 'priorScale', {}, 'gamma', {}, 'mixAlpha', {}, ...
    'bitsPerDatasetCV', {}, 'avgBitsCV', {}, 'medianBitsCV', {}, 'maxBitsCV', {});

bestScore = Inf;
bestMedian = Inf;
best = struct('k', [], 'weightBase', [], 'priorScale', [], 'gamma', [], 'mixAlpha', [], 'shallowK', []);
row = 1;

fprintf('\n========== GLOBAL TRAINING-ONLY TUNING ==========');
fprintf('\nGrid size: %d combinations\n', totalCombos);

for k = ks
    for wb = weightBases
        for ps = priorScales
            for gamma = gammas
                for mixAlpha = mixAlphas
                    comboIdx = comboIdx + 1;
                    bitsPerDataset = zeros(1, numel(datasets));
                    for di = 1:numel(datasets)
                        bitsPerDataset(di) = expmodelTrieTrainCV(datasets{di}, k, wb, ps, gamma, 0.2, mixAlpha);
                    end
                    avgBits = mean(bitsPerDataset);
                    medianBits = median(bitsPerDataset);
                    maxBits = max(bitsPerDataset);
                    results(row).k = k;
                    results(row).weightBase = wb;
                    results(row).priorScale = ps;
                    results(row).gamma = gamma;
                    results(row).mixAlpha = mixAlpha;
                    results(row).bitsPerDatasetCV = bitsPerDataset;
                    results(row).avgBitsCV = avgBits;
                    results(row).medianBitsCV = medianBits;
                    results(row).maxBitsCV = maxBits;
                    fprintf('  [%d/%d] k=%d wb=%.1f ps=%.2f g=%.1f mix=%.2f => avg=%.4f median=%.4f max=%.4f\n', ...
                        comboIdx, totalCombos, k, wb, ps, gamma, mixAlpha, avgBits, medianBits, maxBits);
                    if avgBits < bestScore || (abs(avgBits - bestScore) < 1e-12 && medianBits < bestMedian)
                        bestScore = avgBits;
                        bestMedian = medianBits;
                        best.k = k;
                        best.weightBase = wb;
                        best.priorScale = ps;
                        best.gamma = gamma;
                        best.mixAlpha = mixAlpha;
                        best.shallowK = defaultShallowK(k);
                    end
                    row = row + 1;
                end
            end
        end
    end
end

if isempty(results)
    error('No successful global tuning combinations were found.');
end

[~, order] = sortrows([[results.avgBitsCV]' [results.medianBitsCV]' [results.maxBitsCV]']);
results = results(order);
best = results(1);
best.shallowK = defaultShallowK(best.k);

globalDefaultParams = struct( ...
    'k', best.k, ...
    'weightBase', best.weightBase, ...
    'priorScale', best.priorScale, ...
    'gamma', best.gamma, ...
    'mixAlpha', best.mixAlpha, ...
    'shallowK', best.shallowK, ...
    'avgBitsCV', best.avgBitsCV, ...
    'medianBitsCV', best.medianBitsCV, ...
    'maxBitsCV', best.maxBitsCV, ...
    'bitsPerDatasetCV', best.bitsPerDatasetCV);

save('globalDefaultParams.mat', 'globalDefaultParams', 'results', 'datasets', 'benchmarks');

fprintf('\nBest global default params:\n');
fprintf('  k=%d wb=%.2f ps=%.3f g=%.1f mix=%.2f\n', ...
    globalDefaultParams.k, globalDefaultParams.weightBase, globalDefaultParams.priorScale, ...
    globalDefaultParams.gamma, globalDefaultParams.mixAlpha);
fprintf('  avgBitsCV=%.4f medianBitsCV=%.4f maxBitsCV=%.4f\n', ...
    globalDefaultParams.avgBitsCV, globalDefaultParams.medianBitsCV, globalDefaultParams.maxBitsCV);
end

function shallowK = defaultShallowK(k)
shallowK = min(2, max(1, k - 1));
end
