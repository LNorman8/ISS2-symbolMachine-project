% tuneExpmodelTrie.m
% Grid search for global expmodelTrie hyperparameters across all benchmark datasets.

datasets = {'Hawaiian','Dickens','ElecDemand','DIAtemp','DIAwind','solarWind','HoustonRain'};

ks = [5, 6, 7];
weightBases = [2.0, 2.2, 2.4];
priorScales = [0.1, 0.15, 0.2, 0.25];
gammas = [2.0, 3.0, 4.0, 5.0];

results = struct('k',{},'weightBase',{},'priorScale',{},'gamma',{},'bitsPerDataset',{},'avgBits',{},'medianBits',{},'maxBits',{});
row = 1;

for k = ks
    for wb = weightBases
        for ps = priorScales
            for g = gammas
                fprintf('Testing k=%d wb=%.1f ps=%.2f g=%.1f\n', k, wb, ps, g);
                bitsPerDataset = nan(size(datasets));
                success = true;
                for di = 1:numel(datasets)
                    name = datasets{di};
                    try
                        bitsPerDataset(di) = expmodelTrie(name, 'autoTune', false, 'k', k, 'weightBase', wb, 'priorScale', ps, 'gamma', g);
                    catch ME
                        fprintf('  Failed on %s: %s\n', name, ME.message);
                        success = false;
                        break;
                    end
                end
                if ~success
                    continue;
                end
                avgBits = mean(bitsPerDataset);
                medianBits = median(bitsPerDataset);
                maxBits = max(bitsPerDataset);
                results(row).k = k;
                results(row).weightBase = wb;
                results(row).priorScale = ps;
                results(row).gamma = g;
                results(row).bitsPerDataset = bitsPerDataset;
                results(row).avgBits = avgBits;
                results(row).medianBits = medianBits;
                results(row).maxBits = maxBits;
                fprintf('  avg=%.4f median=%.4f max=%.4f\n', avgBits, medianBits, maxBits);
                row = row + 1;
            end
        end
    end
end

if isempty(results)
    error('No successful parameter combinations were found.');
end

% Sort by average bits, then median bits.
[~, order] = sortrows([[results.avgBits]' [results.medianBits]']);
results = results(order);

best = results(1);

fprintf('\nBest global hyperparameters across all datasets:\n');
fprintf('  k=%d weightBase=%.1f priorScale=%.2f gamma=%.1f\n', best.k, best.weightBase, best.priorScale, best.gamma);
fprintf('  avgBits=%.4f medianBits=%.4f maxBits=%.4f\n', best.avgBits, best.medianBits, best.maxBits);
for di = 1:numel(datasets)
    fprintf('  %s: %.4f\n', datasets{di}, best.bitsPerDataset(di));
end

save('tuneExpmodelTrie_results.mat','results','best','datasets');
