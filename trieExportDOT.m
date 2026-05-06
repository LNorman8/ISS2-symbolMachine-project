function trieExportDOT(trieChildren, trieCounts, maxDepth, nodeBudget, dotFile)
%TRIEEXPORTDOT Export a pruned trie subset to Graphviz DOT format.
if nargin < 3 || isempty(maxDepth), maxDepth = 6; end
if nargin < 4 || isempty(nodeBudget), nodeBudget = 5000; end
if nargin < 5 || isempty(dotFile), dotFile = 'trie_export.dot'; end

nodeCount = size(trieChildren,1);
included = false(nodeCount,1);
included(1) = true;
depths = zeros(nodeCount,1);

queue = 1;
while ~isempty(queue) && nnz(included) < nodeBudget
    node = queue(1); queue(1) = [];
    if depths(node) >= maxDepth
        continue;
    end
    for s = 1:9
        child = double(trieChildren(node,s));
        if child > 0 && ~included(child)
            included(child) = true;
            depths(child) = depths(node) + 1;
            queue(end+1) = child; %#ok<AGROW>
        end
    end
end

fid = fopen(dotFile,'w');
if fid == -1
    error('Could not open %s for writing', dotFile);
end
fprintf(fid, 'digraph Trie {\n');
fprintf(fid, '  node [shape=box, fontsize=10];\n');

origIds = find(included);
for i = 1:numel(origIds)
    n = origIds(i);
    cnt = sum(trieCounts(n,:));
    lbl = sprintf('id=%d\\nD=%d\\nC=%d', n, depths(n), cnt);
    fprintf(fid, '  n%d [label="%s"];\n', n, lbl);
end

for n = origIds(:)'
    for s = 1:9
        c = double(trieChildren(n,s));
        if c > 0 && included(c)
            fprintf(fid, '  n%d -> n%d [label="%d"];\n', n, c, s);
        end
    end
end

fprintf(fid, '}\n');
fclose(fid);
end
