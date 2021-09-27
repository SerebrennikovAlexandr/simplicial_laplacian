import numpy as np

from itertools import permutations
from sympy.combinatorics.permutations import Permutation
from ._checkers import _check_n_simpl_complex, _check_for_laplace, _check_weights, _check_integer_values


# =========== ТЕХНИЧЕСКИЕ ФУНКЦИИ ===========


def _get_bo_matrix_without_permutation(simplex_list, prev_simplex_list, k, orient):
    len_simplex_list = len(simplex_list)
    len_prev_simplex_list = len(prev_simplex_list)

    res = np.zeros((len_prev_simplex_list, len_simplex_list))

    for i, simplex in enumerate(simplex_list):
        sign = orient

        for j in range(k + 1):
            idx = np.ones(k + 1).astype(bool)
            idx[j] = False
            prev_simplex = simplex[idx]

            ind = np.flatnonzero(np.equal(prev_simplex_list, prev_simplex).all(axis=1))[0]
            res[ind][i] = sign
            sign = -sign

    return res


# ============ ОСНОВНЫЕ ФУНКЦИИ ============


def boundary_operator_matrix(n_complex, k=1, p=1, orient=1):
    _check_n_simpl_complex(n_complex)
    k, p = _check_integer_values(k=k, p=p)
    if k < 1:
        raise ValueError('Incorrect value of k: must be >= 1')
    if p < 1:
        raise ValueError('Incorrect value of p: must be >= 1')
    if k - p < 0:
        raise ValueError('Incorrect value of k and p: k - p must be >= 0')
    if orient not in [-1, 1]:
        raise ValueError('Incorrect value of orient: must be 1 or -1')

    simplex_list = n_complex[k]
    prev_simplex_list = n_complex[k - p]

    len_simplex_list = len(simplex_list)
    len_prev_simplex_list = len(prev_simplex_list)

    if len_prev_simplex_list == 0:
        raise ValueError('No ' + str(k - p) + '-simplices were found in the complex')
    elif len_simplex_list == 0:
        raise ValueError('No ' + str(k) + '-simplices were found in the complex')

    if p == 1:
        return _get_bo_matrix_without_permutation(simplex_list, prev_simplex_list, k, orient)

    res = np.zeros((len_prev_simplex_list, len_simplex_list))

    for i, simplex in enumerate(simplex_list):
        mask = np.ones(k + 1).astype(bool)
        mask[:p] = False

        for idx in np.array(sorted(set(permutations(mask)))):
            prev_simplex = simplex[idx]

            vertices_in = list(np.array(range(k + 1))[idx])
            vertices_out = list(np.array(range(k + 1))[~idx])
            sign = orient * Permutation(vertices_out + vertices_in).signature()

            ind = np.flatnonzero(np.equal(prev_simplex_list, prev_simplex).all(axis=1))[0]
            res[ind][i] = sign

    return res


def laplace_matrix(n_complex, k, p=1, q=1, orient=1):
    k, p, q = _check_for_laplace(n_complex, k, p, q, orient)

    if k - p == -1:
        bound_op = boundary_operator_matrix(n_complex, k=k + q, p=q, orient=orient)
        res = np.dot(bound_op, bound_op.T)
    else:
        bound_op1 = boundary_operator_matrix(n_complex, k=k, p=p, orient=orient)
        bound_op2 = boundary_operator_matrix(n_complex, k=k + q, p=q, orient=orient)
        res = np.dot(bound_op1.T, bound_op1) + np.dot(bound_op2, bound_op2.T)

    return res


def weighted_laplace_matrix(n_complex, weights, k, p=1, q=1, orient=1):
    k, p, q = _check_for_laplace(n_complex, k, p, q, orient)
    _check_weights(weights)

    if k - p == -1:
        bom = boundary_operator_matrix(n_complex, k=k + q, p=q, orient=orient)
        w_k_inv = np.linalg.inv(weights[k])
        w_kq = weights[k + q]
        res = w_k_inv @ bom @ w_kq @ bom.T
    else:
        bom_1 = boundary_operator_matrix(n_complex, k=k, p=p, orient=orient)
        bom_2 = boundary_operator_matrix(n_complex, k=k + q, p=q, orient=orient)
        w_kp_inv = np.linalg.inv(weights[k - p])
        w_k = weights[k]
        w_k_inv = np.linalg.inv(w_k)
        w_kq = weights[k + q]
        res = bom_1.T @ w_kp_inv @ bom_1 @ w_k + w_k_inv @ bom_2 @ w_kq @ bom_2.T

    return res
