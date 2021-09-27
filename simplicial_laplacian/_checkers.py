import numpy as np


# ==== ФУНКЦИИ ДЛЯ ПРОВЕРКИ КОРРЕКТНОСТИ ВВОДА ====


def _check_simplex(simplex):
    if not isinstance(simplex, (np.ndarray)) or simplex.ndim != 1 or np.array(simplex).shape[0] == 0:
        t = str(type(simplex))
        raise TypeError('Incorrect type of simplex: given ' + t + '; must be non-empty 1d numpy.ndarray')


def _check_n_simpl_complex(n_complex):
    if not isinstance(n_complex, list):
        raise TypeError('Incorrect type of n_complex: must be list of np.ndarrays of simplices')
    for i in range(len(n_complex)):
        if not isinstance(n_complex[i], np.ndarray):
            raise TypeError('Incorrect type of n_complex: must be list of np.ndarrays of simplices')
        for simplex in n_complex[i]:
            _check_simplex(simplex)


def _check_adjacency_matrix(X):
    if not isinstance(X, (np.ndarray)) or X.ndim != 2 or X.shape[0] != X.shape[1]:
        raise TypeError('Incorrect type of adjacency matrix X: must be numpy.ndarray of shape V x V')


def _check_weights(weights):
    if not isinstance(weights, list):
        raise TypeError('Incorrect type of weights: must be list of diagonal matrices (np.ndarrays)')
    for i in range(len(weights)):
        if not isinstance(weights[i], np.ndarray):
            raise TypeError('Incorrect type of weights: must be list of diagonal matrices (np.ndarrays)')


def _check_integer_values(**kwarg):
    out = []
    for name, val in kwarg.items():
        try:
            out.append(int(val))
        except:
            raise TypeError('Incorrect type of ' + name + ': must be integer')

    if len(out) > 1:
        return tuple(out)
    else:
        return out[0]


def _check_for_laplace(n_complex, k, p, q, orient):
    _check_n_simpl_complex(n_complex)
    k, p, q = _check_integer_values(k=k, p=p, q=q)
    if k < 0:
        raise ValueError('Incorrect value of k: must be >= 0')
    if p < 1:
        raise ValueError('Incorrect value of p: must be >= 1')
    if k - p < -1:
        raise ValueError('Incorrect value of k and p: k - p must be >= 0')
    if q < 1:
        raise ValueError('Incorrect value of q: must be >= 1')
    if orient not in [-1, 1]:
        raise ValueError('Incorrect value of orient: must be 1 or -1')
    return k, p, q
