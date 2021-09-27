import numpy as np

from itertools import permutations
from ._checkers import _check_simplex, _check_integer_values, _check_adjacency_matrix


# =========== ТЕХНИЧЕСКИЕ ФУНКЦИИ ===========


def _neighbors_lower(x_id, A):
    LT = np.tril(A)
    return list(np.nonzero(LT[x_id])[0])


def _intersect(d):
    return list(set(d[0]).intersection(*d))


def _grade(K):
    dim = len(max(K, key=len))
    K_graded = [[] for _ in range(dim)]

    for sigma in K:
        dim_sigma = len(sigma) - 1

        if dim_sigma == 0:
            sigma = sigma[0]
        K_graded[dim_sigma].append(sigma)

    for k, item in enumerate(K_graded):
        if k == 0:
            item_array = np.expand_dims(np.array(item), 1)
        else:
            item_array = np.array(item)

        K_graded[k] = item_array

    return K_graded


# ==== ФУНКЦИИ ДЛЯ ПОСТРОЕНИЯ СИМПЛИЦИАЛЬНЫХ КОМПЛЕКСОВ ====


def subsimplex(simplex, diff=1):
    _check_simplex(simplex)
    diff = _check_integer_values(diff=diff)
    if diff < 0:
        raise ValueError('Incorrect diff value: must be >= 0')

    n = len(simplex)
    res = []

    mask = np.ones(n).astype(bool)
    mask[:diff] = False
    for idx in np.array(sorted(set(permutations(mask)))):
        sub_simplex = np.array(simplex)[idx]
        res.append(sub_simplex)

    res = np.sort(res, axis=0)
    return res


def simplicial_complex_vietoris_rips(X, n_skeleton):
    _check_adjacency_matrix(X)
    n_skeleton = _check_integer_values(n_skeleton=n_skeleton)
    if n_skeleton < 0:
        raise ValueError('Incorrect value of n_skeleton: must be >= 0')

    n_skeleton = n_skeleton + 1

    def add_cofaces(A, k, tau, N_lower, simplices):

        simplices.append(tau)

        if len(tau) >= k:
            return

        else:
            for v in N_lower:
                sigma = sorted(tau + [v])
                M = _intersect([N_lower, _neighbors_lower(v, A)])
                add_cofaces(A, k, sigma, M, simplices)

        return simplices

    simplices = []

    V = list(range(X.shape[0]))

    for u in V:
        N_lower = _neighbors_lower(u, X)
        add_cofaces(X, n_skeleton, [u], N_lower, simplices)

    res = _grade(simplices)

    for i in range(len(res)):
        res[i] = np.sort(np.unique(np.array(res[i]), axis=0))

    while len(res) < n_skeleton:
        res.append(np.array([]))

    return res


def simplicial_complex(X, n_skeleton=2, weighted=False):
    _check_adjacency_matrix(X)
    n_skeleton = _check_integer_values(n_skeleton=n_skeleton)
    if n_skeleton < 0:
        raise ValueError('Incorrect value of n_skeleton: must be >= 0')

    N, res, res_weights = X.shape[0], [[]], [np.eye(X.shape[0])]

    for i in range(N):
        res[0].append([i])

    res[0] = np.array(res[0])

    for iteration in range(1, n_skeleton + 1):
        res.append([])
        prev_simplex_list = res[iteration - 1]

        for simplex in prev_simplex_list:
            for i in range(N):
                flag = True

                for j in simplex:
                    if X[i][j] == 0:
                        flag = False
                        break

                if flag:
                    new_clique = np.append(simplex, i)
                    new_clique.sort()
                    res[iteration].append(new_clique)

        res[iteration] = np.sort(np.unique(np.array(res[iteration]), axis=0))

        if weighted:
            n_simplices = res[iteration].shape[0]
            res_weights.append(np.zeros((n_simplices, n_simplices)))

            for i, simplex in enumerate(res[iteration]):
                if iteration == 1:
                    res_weights[1][i][i] = X[simplex[0]][simplex[1]]

                else:
                    weight = -np.inf

                    for prev_simplex in subsimplex(simplex):
                        ind = np.flatnonzero(np.equal(prev_simplex_list, prev_simplex).all(axis=1))[0]
                        prev_simplex_weight = res_weights[iteration - 1][ind][ind]
                        weight = np.maximum(weight, prev_simplex_weight)

                    res_weights[iteration][i][i] = weight

    if weighted:
        return res, res_weights
    return res
