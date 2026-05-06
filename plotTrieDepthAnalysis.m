function plotTrieDepthAnalysis(trieChildren, trieCounts, nodeCount, k, outputFile)
%PLOTTRIEDEPTHANALYSIS Plot node counts and branching statistics by depth.
%   plotTrieDepthAnalysis(trieChildren, trieCounts, nodeCount, k, outputFile)

if nargin < 4
    error('plotTrieDepthAnalysis requires trieChildren, trieCounts, nodeCount, and k');
end
if nargin < 5
    outputFile = '';
end

nodeCount = double(nodeCount);
depths = zeros(nodeCount,1);
depths(1) = 0;
q = 1;
head = 1;
while head <= numel(q)
    node = q(head); head = head + 1;
    for s = 1:9
        child = double(trieChildren(node,s));
        if child > 0 && depths(child) == 0 && child ~= 1
            depths(child) = depths(node) + 1;
            q(end+1) = child; %#ok<AGROW>
        end
    end
end

maxDepth = max(depths);
nodesPerDepth = zeros(maxDepth+1,1);
avgBranch = zeros(maxDepth+1,1);
for d = 0:maxDepth
    idx = find(depths == d);
    nodesPerDepth(d+1) = numel(idx);
    if nodesPerDepth(d+1) > 0
        outEdges = 0;
        for ni = idx(:)'
            outEdges = outEdges + nnz(trieChildren(ni,:));
        end
        avgBranch(d+1) = outEdges / nodesPerDepth(d+1);
    else
        avgBranch(d+1) = 0;
    end
end

figure('Name','Trie Depth Analysis','NumberTitle','off');
yyaxis left
bar(0:maxDepth, nodesPerDepth, 'FaceColor',[0.2 0.6 0.8]);
xlabel('Depth'); ylabel('Nodes');
yyaxis right
plot(0:maxDepth, avgBranch, '-o', 'Color',[0.85 0.33 0.10], 'LineWidth',1.5);
ylabel('Avg branching (children per node)');
title(sprintf('Trie depth stats (max k=%d) — nodes=%d', k, nodeCount));
grid on;

if ~isempty(outputFile)
    try
        saveas(gcf, outputFile);
    catch
        warning('Could not save %s', outputFile);
    end
end
end
