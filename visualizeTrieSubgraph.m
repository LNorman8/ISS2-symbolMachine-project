function visualizeTrieSubgraph(trieChildren, trieCounts, maxDepth, nodeBudget, outputFile)
%VISUALIZETRIESUBGRAPH Render a pruned trie subgraph and save to PNG if requested.
if nargin < 3 || isempty(maxDepth), maxDepth = 6; end
if nargin < 4 || isempty(nodeBudget), nodeBudget = 2000; end
if nargin < 5, outputFile = ''; end

[G, nodeMap, meta] = trieToDigraph(trieChildren, trieCounts, maxDepth, nodeBudget);
if numnodes(G) == 0
    warning('No nodes in the selected trie subgraph.');
    return;
end

h = figure('Name','Trie Subgraph','NumberTitle','off','Visible','on');
p = plot(G, 'Layout','layered', 'EdgeLabel', G.Edges.Symbol);
title(sprintf('Trie subgraph (maxDepth=%d, nodes=%d)', meta.maxDepth, numnodes(G)));

% Annotate nodes with (origId, depth)
labels = cell(numnodes(G),1);
for i = 1:numnodes(G)
    orig = meta.origIds(i);
    d = meta.depths(orig);
    count = sum(trieCounts(orig,:));
    labels{i} = sprintf('%d\nD=%d\nC=%d', orig, d, count);
end
labelnode(p, 1:numnodes(G), labels);

if ~isempty(outputFile)
    try
        saveas(h, outputFile);
    catch
        warning('Could not save %s', outputFile);
    end
end
end
