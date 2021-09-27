import numpy as np
import networkx as nx

from tqdm import tqdm
from ._checkers import _check_integer_values


# ============== ФУНКЦИИ ДЛЯ ПОЛУЧЕНИЯ ДАННЫХ ==============


def TUDataset(ds_name, weighted=False, verbose=True, n_first=None):
    if not isinstance(ds_name, str):
        raise TypeError('ds_name - is the name of a dataset, must be str')
    if n_first is not None:
        n_first = _check_integer_values(n_first=n_first)

    if verbose:
        with open('{}/README.txt'.format(ds_name), 'r') as fin:
            for line in fin.readlines():
                print(line.strip())

    res_y = []
    res_adj_matrix = []

    graph_indicator = []
    edges = []

    with open('{}/{}_graph_labels.txt'.format(ds_name, ds_name), 'r') as fin:
        for line in fin.readlines():
            res_y.append(int(line.strip()))
    res_y = np.array(res_y)
    if n_first is not None:
        res_y = res_y[:n_first]

    with open('{}/{}_graph_indicator.txt'.format(ds_name, ds_name), 'r') as fin:
        for line in fin.readlines():
            graph_indicator.append(int(line.strip()))

    with open('{}/{}_A.txt'.format(ds_name, ds_name), 'r') as fin:
        for line in fin.readlines():
            edges.append(list(map(int, line.strip().split(', '))))

    graph_indicator = np.array(graph_indicator)
    edges = np.array(edges)

    if weighted:
        edge_weights = []
        with open('{}/{}_edge_attributes.txt'.format(ds_name, ds_name), 'r') as fin:
            for line in fin.readlines():
                edge_weights.append(float(line.strip()))
        edge_weights = np.array(edge_weights)

    left = 0
    right = 1
    n_edges = 1

    for i in range(1, res_y.shape[0] + 1):
        left += n_edges
        n_edges = (graph_indicator == i).sum()
        right += n_edges

        cur_adj_matrix = np.zeros((n_edges, n_edges))

        idx = (edges[:, 0] >= left) & (edges[:, 0] < right)
        cur_edges = edges[idx]
        if weighted:
            cur_edges_weights = edge_weights[idx]

        for j, cur_edge in enumerate(cur_edges):
            v1 = cur_edge[0] - left
            v2 = cur_edge[1] - left
            if weighted:
                cur_adj_matrix[v1][v2] = cur_edges_weights[j]
            else:
                cur_adj_matrix[v1][v2] = 1

        res_adj_matrix.append(cur_adj_matrix)

    res_y = np.array(res_y)

    return res_adj_matrix, res_y


def watts_strogatz(N_graphs, n_nodes, k_nearest_neighbours, probability_list):
    for i, ps in enumerate(probability_list):
        probability_list[i] = _check_integer_values(probability_i=ps)
    if len(probability_list) == 0:
        raise ValueError("Length of probability_list is 0")
    N, n, k = _check_integer_values(N_graphs=N_graphs, n_nodes=n_nodes,
                                    k_nearest_neighbours=k_nearest_neighbours)

    res_graphs = []
    res_g_y = []

    for p_i in tqdm(np.repeat(probability_list, N)):
        G = nx.generators.random_graphs.watts_strogatz_graph(n, k, p_i)
        res_graphs.append(nx.to_numpy_array(G))

    for i in range(len(probability_list)):
        res_g_y += [i] * N
    res_g_y = np.array(res_g_y)

    return res_graphs, res_g_y
