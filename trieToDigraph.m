function [G, nodeMap, meta] = trieToDigraph(trieChildren, trieCounts, maxDepth, nodeBudget)
%PRIETODIGRAPH Convert a pruned subset of trie into a MATLAB digraph.
%   [G, nodeMap, meta] = trieToDigraph(trieChildren, trieCounts, maxDepth, nodeBudget)

if nargin < 3 || isempty(maxDepth)
    maxDepth = 6;
end
if nargin < 4 || isempty(nodeBudget)
    nodeBudget = 5000;
end

nodeCount = size(trieChildren,1);
depths = zeros(nodeCount,1);
included = false(nodeCount,1);
included(1) = true;
depths(1) = 0;

edgesFrom = [];
edgesTo = [];
edgeLabel = [];

queue = 1;
while ~isempty(queue) && nnz(included) < nodeBudget
    node = queue(1); queue(1) = [];
    if depths(node) >= maxDepth
        continue;
    end
    for s = 1:9
        child = double(trieChildren(node,s));
        if child > 0
            if ~included(child)
                included(child) = true;
                depths(child) = depths(node) + 1;
                queue(end+1) = child; %#ok<AGROW>
            end
            edgesFrom(end+1) = node; %#ok<AGROW>
            edgesTo(end+1) = child; %#ok<AGROW>
            edgeLabel{end+1,1} = sprintf('%d', s); %#ok<AGROW>
        end
    end
end

% Map original node ids to compact ids for digraph
origIds = find(included);
newId = zeros(nodeCount,1);
newId(origIds) = 1:numel(origIds);

if isempty(edgesFrom)
    G = digraph(); nodeMap = newId; meta = struct('depths',depths,'origIds',origIds); return;
end

u = newId(edgesFrom);
v = newId(edgesTo);

% Aggregate labels for any duplicate edges (same u->v)
T = table(u(:), v(:), edgeLabel(:), 'VariableNames', {'U','V','Symbol'});
[uv, ~, ic] = unique([T.U T.V], 'rows', 'stable');
labelsAgg = cell(size(uv,1),1);
for jj = 1:size(uv,1)
    labelsAgg{jj} = strjoin(T.Symbol(ic == jj), ',');
end

G = digraph(uv(:,1), uv(:,2));
G.Edges.Symbol = labelsAgg;

nodeMap = newId; % index by original node -> new node id (0 = excluded)
meta = struct('depths', depths, 'origIds', origIds, 'nodeBudget', nodeBudget, 'maxDepth', maxDepth);
end
