clear;clc;


%%%Parameters%%
n_nodes = 20;
n_graphes = 1000;
density = rand*0.5;
test_mode = true;
%%%%%%%%%%%%%%%


path = './GC_20_D_1to5/';

if test_mode
    [A,Aj] = random_connected_graph(n_nodes, density);
    plot(graph(Aj));    
else
    for i = 1:n_graphes
        [A,Aj] = random_connected_graph(n_nodes, density);
        
        % perform reordering algos
        p_nested = dissect(A);
        p_AMD = amd(A);
        p_RCM = symrcm(A);
        
        % choose the best
        optimal_order = compare(A, p_nested, p_AMD, p_RCM);
        % node id starts from 0
        optimal_order = optimal_order - 1;
        
        % write into csv
        status = write_result(A, optimal_order, i, path);
        disp(status);
    end
end


function [A, Aj] = random_connected_graph(n_nodes, density)
    connectivity = 10;
    disp('Searching...')
    while connectivity ~= 1
        % generate a sparse postive matrix
        shape = randi([n_nodes, n_nodes]);
        density = density + 0.01*rand(1,1);
        A = random_PSD(shape, density);
        
        % compute it's ajacency matrix
        ones_ = diag(ones(1, size(A,1)));
        mask = ones_ - 1;
        Aj = A.*mask;
        Aj = Aj ~= 0;
        
        % compute it's laplacian matrix
        Al = -1*Aj + diag(sum(Aj));
        
        % check the number of 0 eigen values
        connectivity = nnz(~round(eig(Al),5));
    end
    disp('Done')
end



function sample = random_PSD(s, d)
    % s: size
    % d: density: entries = density*s*s
    sample = sprand(s,s,d);
    % ensuring PD
    sample = sample + sample';
    d = abs(sum(sample)) + 1;
    sample = sample + diag(sparse(d));
end



function order = compare(A, p_nested, p_amd, p_rcm)

    A_nested = A(p_nested, p_nested);
    nnz_chol_nested = nnz(chol(A_nested));

    A_amd = A(p_amd, p_amd);
    nnz_chol_amd = nnz(chol(A_amd));

    A_rcm = A(p_rcm, p_rcm);
    nnz_chol_rcm = nnz(chol(A_rcm));

    if nnz_chol_amd < nnz_chol_rcm && nnz_chol_amd < nnz_chol_nested
%         optimal = 'AMD';
        order = p_amd;
    elseif nnz_chol_nested < nnz_chol_rcm && nnz_chol_nested < nnz_chol_amd
%         optimal = 'Dissct';
        order = p_nested;
    else
%         optimal = 'RCM';
        order = p_rcm;
    end
end


function status = write_result(matrix, optimal_order, g_id, path)
    % transform problem into the related ajacent matrix
    A = matrix - diag(diag(matrix));
    G = graph(A);
    
    
    edges = G.Edges.EndNodes;
    % index starts from 0     
    edges = edges - 1;
    graph_id = repelem(g_id, size(edges, 1));
    graph_id = graph_id';
    T = [graph_id, edges];
    T = array2table(T, 'VariableNames',{'graph_id', 'source_node_id','destination_node_id'});

    edges_path = strcat(path,'edges.csv');
    writetable(T, edges_path,'Delimiter',',','WriteMode','append');

   
    graph_id = repelem(g_id, size(matrix,1));
    graph_id = graph_id';
    % node_id = 1:size(matrix,1);
    % node_id = node_id - 1;
    % node_id = node_id';
    % update on 24th June
    % The arg:optimal order is the node id ordered by optima ordering, not
    % ordering! (sort vs. agrsort)
    node_id = optimal_order';
    node_label = 1:size(matrix,1);
    node_label = node_label-1;
    node_label = node_label';
    T = table(graph_id, node_id, node_label);
    T = sortrows(T,2);
    nodes_path = strcat(path,'nodes.csv');
    writetable(T, nodes_path, 'Delimiter',',', 'WriteMode','append');

    status = strcat('Recording graph','  ', int2str(g_id));
end
