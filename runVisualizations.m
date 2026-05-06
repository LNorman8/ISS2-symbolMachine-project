function runVisualizations(name, k, maxDepth, nodeBudget, outputDir)
%RUNVISUALIZATIONS Driver to build trie and create visualizations.
% Usage: runVisualizations('Hawaiian', 8, 5, 2000, 'viz_out')

if nargin < 1 || isempty(name), error('Provide dataset name (e.g., ''Hawaiian'')'); end
if nargin < 2 || isempty(k), k = 6; end
if nargin < 3 || isempty(maxDepth), maxDepth = min(k,6); end
if nargin < 4 || isempty(nodeBudget), nodeBudget = 2000; end
if nargin < 5 || isempty(outputDir), outputDir = fullfile(pwd, 'viz_out'); end

if ~exist(outputDir, 'dir'), mkdir(outputDir); end

trainFile = fullfile(pwd, sprintf('sequence_%s_train.mat', name));
if ~exist(trainFile, 'file')
    error('Train file not found: %s', trainFile);
end

fprintf('Building trie from %s (k=%d)...\n', trainFile, k);
[trieChildren, trieCounts, nodeCount, priorProb, trainSeq] = buildTrieFromTrain(trainFile, k);

fprintf('Plotting depth analysis...\n');
depthOut = fullfile(outputDir, sprintf('depth_stats_%s.png', name));
plotTrieDepthAnalysis(trieChildren, trieCounts, nodeCount, k, depthOut);

fprintf('Creating trie subgraph visualization (maxDepth=%d, nodeBudget=%d)...\n', maxDepth, nodeBudget);
subOut = fullfile(outputDir, sprintf('trie_subgraph_%s.png', name));
visualizeTrieSubgraph(trieChildren, trieCounts, maxDepth, nodeBudget, subOut);

dotOut = fullfile(outputDir, sprintf('trie_%s.dot', name));
fprintf('Exporting DOT to %s\n', dotOut);
trieExportDOT(trieChildren, trieCounts, maxDepth, nodeBudget, dotOut);

fprintf('Saved outputs to %s\n', outputDir);
end
