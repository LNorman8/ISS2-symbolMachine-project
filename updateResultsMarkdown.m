function updateResultsMarkdown()
% Regenerate results.md from saved tuning artifacts.

load('perClassParams.mat', 'datasets', 'benchmarks');
currentParams = [];
if isfile('datasetParams.mat')
    data = load('datasetParams.mat');
    if isfield(data, 'datasetParams')
        currentParams = data.datasetParams;
    end
end
modelLimits = [];
if isfile('modelLimits.mat')
    data = load('modelLimits.mat');
    if isfield(data, 'modelLimits')
        modelLimits = data.modelLimits;
    end
end

lines = {};
lines{end+1} = '# Results';
lines{end+1} = '';
lines{end+1} = '|Dataset    |Benchmark|Current Status|Model Limit| K | weightBase | priorScale | gamma | mixAlpha |';
lines{end+1} = '|-----------|---------|--------------|-----------|---|------------|------------|-------|----------|';

for i = 1:numel(datasets)
    benchmark = benchmarks(i);
    currentBits = NaN;
    params = struct('k', NaN, 'weightBase', NaN, 'priorScale', NaN, 'gamma', NaN, 'mixAlpha', NaN);
    if ~isempty(currentParams)
        idx = find(strcmp({currentParams.name}, datasets{i}), 1);
        if ~isempty(idx)
            currentBits = currentParams(idx).testBits;
            params = currentParams(idx).params;
        end
    end
    modelLimitBits = NaN;
    if ~isempty(modelLimits)
        idx = find(strcmp({modelLimits.name}, datasets{i}), 1);
        if ~isempty(idx)
            modelLimitBits = modelLimits(idx).bitsPerSymbol;
        end
    end

    lines{end+1} = sprintf('|%-11s| %0.4f  | %0.4f       | %0.4f    | %d | %0.2f       | %0.3f      | %0.1f   | %0.2f     |', ...
        datasets{i}, benchmark, currentBits, modelLimitBits, params.k, params.weightBase, params.priorScale, params.gamma, params.mixAlpha);
end

lines{end+1} = '';
lines{end+1} = 'Model limit is computed by running the model with the test sequence used as the training sequence as well, so it is an oracle upper bound rather than a valid benchmark result.';

fid = fopen('results.md', 'w');
if fid < 0
    error('Could not open results.md for writing.');
end
for i = 1:numel(lines)
    fprintf(fid, '%s\n', lines{i});
end
fclose(fid);

fprintf('results.md updated.\n');
end
