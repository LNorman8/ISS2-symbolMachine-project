%%% DATASET CLASSIFICATION BY ENTROPY
% Computes Shannon entropy for each training sequence and assigns datasets
% to one of 4 classes: Very Low, Low, Medium, High entropy.
% Saves classification and entropy stats to perClassParams.mat

datasets = {'Hawaiian', 'ElecDemand', 'Dickens', 'DIAtemp', 'DIAwind', 'solarWind', 'HoustonRain'};
benchmarks = [1.6456, 1.3387, 1.7674, 1.6797, 1.8762, 0.6692, 0.1012];

% Compute entropy for each dataset
entropies = zeros(1, 7);
for i = 1:7
    trainFile = ['sequence_' datasets{i} '_train.mat'];
    seq_struct = load(trainFile);
    sequence = seq_struct.sequence(:)';

    % Compute Shannon entropy
    counts = histcounts(sequence, 0.5:1:9.5);
    probs = counts / sum(counts);
    probs(probs == 0) = [];
    entropies(i) = -sum(probs .* log2(probs));
end

% Display entropy statistics
fprintf('\n========== DATASET ENTROPY CLASSIFICATION ==========\n');
fprintf('Dataset\t\tEntropy\tBenchmark\n');
fprintf('-------\t\t-------\t---------\n');
for i = 1:7
    fprintf('%s\t\t%.4f\t%.4f\n', datasets{i}, entropies(i), benchmarks(i));
end

% Classify into 4 entropy regimes
% Thresholds: 0.5, 1.5, 2.5 bits (adjusted based on actual data distribution)
classNames = {'VeryLowEntropy', 'LowEntropy', 'MediumEntropy', 'HighEntropy'};
classThresholds = [0, 0.5, 1.5, 2.5, Inf];  % Lower and upper bounds

% Assign each dataset to a class
datasetClass = zeros(1, 7);
classMembers = cell(1, 4);
for c = 1:4
    classMembers{c} = [];
end

for i = 1:7
    for c = 1:4
        if entropies(i) >= classThresholds(c) && entropies(i) < classThresholds(c+1)
            datasetClass(i) = c;
            classMembers{c} = [classMembers{c}, i];
            break;
        end
    end
end

% Display classification
fprintf('\n========== CLASS ASSIGNMENTS ==========\n');
for c = 1:4
    fprintf('%s (entropy < %.2f):\n', classNames{c}, classThresholds(c+1));
    if ~isempty(classMembers{c})
        for idx = classMembers{c}
            fprintf('  %s (entropy=%.4f)\n', datasets{idx}, entropies(idx));
        end
    else
        fprintf('  (empty)\n');
    end
end

% Save classification and parameters
save('perClassParams.mat', 'datasets', 'benchmarks', 'entropies', 'datasetClass', ...
    'classNames', 'classThresholds', 'classMembers');

fprintf('\nClassification saved to perClassParams.mat\n');
