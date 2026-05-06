function test_visualize_small()
%TEST_VISUALIZE_SMALL Simple test for trie visualization helpers
trainSeq = randi(9, 1, 300);

% For compatibility with buildTrieFromTrain expecting 'sequence' variable name
sequence = trainSeq; %#ok<NASGU>
save('temp_small_train.mat', 'sequence');

k = 4;
[trieChildren, trieCounts, nodeCount, priorProb, ~] = buildTrieFromTrain('temp_small_train.mat', k);
plotTrieDepthAnalysis(trieChildren, trieCounts, nodeCount, k, 'temp_depth.png');
visualizeTrieSubgraph(trieChildren, trieCounts, 3, 1000, 'temp_subgraph.png');
trieExportDOT(trieChildren, trieCounts, 3, 1000, 'temp_trie.dot');

fprintf('Generated temp_depth.png, temp_subgraph.png, temp_trie.dot\n');
end
