import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score
from ._checkers import _check_adjacency_matrix, _check_integer_values, _check_n_simpl_complex
from .laplace import laplace_matrix, weighted_laplace_matrix


# =========== ТЕХНИЧЕСКИЕ ФУНКЦИИ ===========


def _is_subset(lst1, lst2):
    return set(lst1).issubset(set(lst2))


def _contains(row1, row2):
    return _is_subset(row2.nonzero()[0], row1.nonzero()[0])


def _remove_zero_vertices(x):
    idx = x.any(axis=1)
    x = x[:, idx]
    x = x[idx]
    return x


def _is_scaffold(x):
    for i in range(x.shape[0]):
        for j in list(x[i].nonzero()[0]):
            if i != j and _contains(x[j], x[i]):
                return False
    return True


def _cos_sim(a, b):
    return 1 - (np.dot(a, b) / (np.sqrt((a**2).sum()) * np.sqrt((b**2).sum())))


# ============== ФУНКЦИИ ДЛЯ ОБРАБОТКИ ДАННЫХ ==============


def get_graph_shapes(graphs, plot_shapes=True):
    """
        Для списка графов, представленных матрицами смежности, выводит все
        различные размеры матриц и количество матриц соответствующего размера
        Возвращает данные значения в виде словаря, может построить диаграмму
        распределения размеров

        Parameters
        ----------
        graphs:             list
            List of adjacency matrices as numpy arrays
        plot_shapes:        bool
            Target feature for finding intervals

        Returns
        -------
        Number of matrices of each size    -   dict
    """

    for g in graphs:
        _check_adjacency_matrix(g)

    t = {}
    sorted_t = {}
    for g in graphs:
        if g.shape in t.keys():
            t[g.shape] += 1
        else:
            t[g.shape] = 1
    sorted_keys = sorted(t, key=t.get, reverse=True)
    for w in sorted_keys:
        sorted_t[w] = t[w]

    if plot_shapes:
        fig, ax = plt.subplots()

        x = np.arange(1, len(sorted_t.keys()) + 1)
        ax.bar(x, sorted_t.values(), tick_label=list(map(str, sorted_t.keys())), width=0.5)
        fig.set_figwidth(15)
        fig.set_figheight(5)

        plt.xticks(rotation=90)
        plt.show()

    return sorted_t


def filter_graphs_by_size(graphs, g_y, shape, shape_upper=None):
    """
        Позволяет отфильтровать список матриц смежности по их
        размерам согласно указанной верхней и нижней границе

        Parameters
        ----------
        graphs:             list
            List of adjacency matrices as numpy arrays
        g_y:                numpy array
            Graph labels
        shape:              int
            Lower border of size interval
        shape_upper:        int
            Upper border of size interval

        Returns
        -------
        Filtered list of adjacency matrices of proper size and labels    -    list, list
    """

    for g in graphs:
        _check_adjacency_matrix(g)
    if isinstance(shape, tuple):
        shape = shape[0]
    if shape_upper is not None and isinstance(shape_upper, tuple):
        shape_upper = shape_upper[0]

    res_graphs = []
    res_g_y = []

    if shape_upper is None:
        for i in range(len(graphs)):
            if graphs[i].shape[0] >= shape:
                res_graphs.append(graphs[i])
                res_g_y.append(g_y[i])
    else:
        for i in range(len(graphs)):
            if graphs[i].shape[0] >= shape and graphs[i].shape[0] <= shape_upper:
                res_graphs.append(graphs[i])
                res_g_y.append(g_y[i])

    res_g_y = np.array(res_g_y)

    return res_graphs, res_g_y


def get_graphs_minmax_edge_values(graphs):
    """
        Позволяет для набора матриц смежности получить
        минимальный и максимальный вес ребер в наборе

        Parameters
        ----------
        graphs:             list
            List of adjacency matrices as numpy arrays

        Returns
        -------
        Minimal and maximal edge value in graphs list    -    tuple(int, int)
    """

    for g in graphs:
        _check_adjacency_matrix(g)
    if len(graphs) == 0:
        raise ValueError("Length of graphs is 0")

    minn = graphs[0][graphs[0] != 0].min()
    maxx = graphs[0].max()

    for g in graphs:
        if g[g != 0].min() < minn:
            minn = g[g != 0].min()
        if g.max() > maxx:
            maxx = g.max()

    return minn, maxx


def one_graph_density(X):
    """
        Позволяет получить степень плотности одного графа
        (насколько граф близок к полносвязному)

        Parameters
        ----------
        X:              numpy array
            Adjacency matrix as numpy arrays

        Returns
        -------
        Graph density (from 0 to 1)    -    float
    """

    _check_adjacency_matrix(X)

    n = X.shape[0]

    e_max = n * (n - 1) / 2
    e_cur = (X != 0).sum() / 2

    return e_cur / e_max


def graphs_mean_density(graphs):
    """
        Позволяет получить среднюю степень плотности графов в указанном наборе графов
        (насколько граф близок к полносвязному)

        Parameters
        ----------
        graphs:             list
            List of adjacency matrices as numpy arrays

        Returns
        -------
        Graph density (from 0 to 1)    -    float
    """

    summ = 0

    for g in graphs:
        summ += one_graph_density(g)

    return summ / len(graphs)


def threshold_unweighted_apply(X, t, inverse=False):
    """
        Применение порога к невзвешенному графу.
        К графу применяется пороговое значение:
        каждому ребру в графе присваивается вес 1 (граф становится невзвешенным),
        если его вес больше (меньше при inverse=True) параметра t, иначе оно удаляется

        Parameters
        ----------
        X:                  numpy array
            Adjacency matrix as numpy arrays
        t:                  float
            Threshold
        inverse:            bool
            Inverse threshold apply

        Returns
        -------
        Adjacency matrix    -    numpy array
    """

    _check_adjacency_matrix(X)

    if inverse:
        convert_func = lambda x: 0 if abs(x) > t else 1
    else:
        convert_func = lambda x: 1 if abs(x) > t else 0

    X_res = np.array([[convert_func(y) for y in x] for x in X])
    np.fill_diagonal(X_res, 0)
    return X_res


def graphs_threshold_unweighted_apply(graphs, t, inverse=False):
    """
        Применение порога к списку невзвешенных графов.
        К каждому графу из списка применяется пороговое значение:
        каждому ребру в графе присваивается вес 1 (граф становится невзвешенным),
        если его вес больше (меньше при inverse=True) параметра t, иначе оно удаляется

        Parameters
        ----------
        graphs:             list
            List of adjacency matrices as numpy arrays
        t:                  float
            Threshold
        inverse:            bool
            Inverse threshold apply

        Returns
        -------
        List of adjacency matrices    -    list
    """

    res_graphs = []

    for g in graphs:
        res_graphs.append(threshold_unweighted_apply(g, t, inverse=inverse))

    return res_graphs


def threshold_weighted_apply(X, t, inverse=False):
    """
        Применение порога к взвешенному графу.
        К графу применяется пороговое значение:
        каждое ребро в графе остается с прежним весом, если
        его вес больше (меньше при inverse=True) параметра t, иначе оно удаляется

        Parameters
        ----------
        X:                  numpy array
            Adjacency matrix as numpy arrays
        t:                  float
            Threshold
        inverse:            bool
            Inverse threshold apply

        Returns
        -------
        Adjacency matrix    -    numpy array
    """

    _check_adjacency_matrix(X)

    if inverse:
        convert_func = lambda x: 0 if abs(x) > t else x
    else:
        convert_func = lambda x: x if abs(x) > t else 0

    X_res = np.array([[convert_func(y) for y in x] for x in X])
    np.fill_diagonal(X_res, 0)
    return X_res


def graphs_threshold_weighted_apply(graphs, t, inverse=False):
    """
        Применение порога к списку взвешенных графов.
        К каждому графу из списка применяется пороговое значение:
        каждое ребро в графе остается с прежним весом, если
        его вес больше (меньше при inverse=True) параметра t, иначе оно удаляется

        Parameters
        ----------
        graphs:             list
            List of adjacency matrices as numpy arrays
        t:                  float
            Threshold
        inverse:            bool
            Inverse threshold apply

        Returns
        -------
        List of adjacency matrices    -    list
    """

    res_graphs = []

    for g in graphs:
        res_graphs.append(threshold_weighted_apply(g, t, inverse=inverse))

    return res_graphs


def filter_complexes_by_simplex_size(complexes_list, y, size, weights_list=None):
    """
        Позволяет отбирать симплициальные комплексы из списка
        по максимальной размерности симплекса, который есть
        в каждом симплициальном комплексе из набора

        Parameters
        ----------
        complexes_list:     list
            List of simplicial complexes (each is a list of numpy arrays)
        y:                  numpy array
            Complex labels
        size:               int
            Max simplex dimension
        weights_list:       list
            List of weights of simplices of each simplicial complex

        Returns
        -------
        Filtered list of simplicial complexes of proper dimension and labels   -    list, list
    """

    if not isinstance(complexes_list, list):
        raise TypeError('complexes_list must be a list of complexes')
    if not isinstance(y, np.ndarray):
        raise TypeError('y must be a Numpy array')
    size = _check_integer_values(size=size)
    if weights_list is not None:
        if not isinstance(weights_list, list):
            raise TypeError('weights_list must be a list of weights')
        if len(complexes_list) != len(weights_list):
            raise ValueError('complexes_list and weights_list must be the same size')

    res_complexes_list = []
    res_y = []
    res_weights = []

    for i in range(len(complexes_list)):
        if len(complexes_list[i][size]) != 0:
            res_complexes_list.append(complexes_list[i])
            res_y.append(y[i])
            if weights_list is not None:
                res_weights.append(weights_list[i])

    res_y = np.array(res_y)

    if weights_list is not None:
        return res_complexes_list, res_weights, res_y
    return res_complexes_list, res_y


def core_scaffold(x, verbose=False):
    """
        Построение топологического каркаса графа

        Parameters
        ----------
        x:                  numpy array
            Adjacency matrix as numpy arrays
        verbose:            bool
            Output

        Returns
        -------
        Scaffold of a graph   -    numpy array
    """

    _check_adjacency_matrix(x)

    x_res = x.copy()
    np.fill_diagonal(x_res, 1)

    while not _is_scaffold(x_res):
        # for every row, check if it contains other rows
        for i in range(x_res.shape[0]):
            for j in list(x_res[i].nonzero()[0]):
                if i != j and _contains(x_res[j], x_res[i]):
                    if verbose:
                        print("Vertice {} contains vertice {}".format(j, i))
                    x_res[i, :] = 0
                    x_res[:, i] = 0

        x_res = _remove_zero_vertices(x_res)

    np.fill_diagonal(x_res, 0)
    return x_res


def homology_cycles(s_complex, laplacian, threshold=0.01):
    """
        Локализация и интерпретация гармонических циклов,
        характеризующихся собственными векторами Лапласиана

        Parameters
        ----------
        s_complex:          list
            Simplicial complex (list of numpy arrays)
        laplacian:          int or tuple (k, p, q)
            Type of (k,p,q)-Laplacian
        threshold:          float
            Threshold for zero eigenvalues

        Returns
        -------
        Array of cycle characteristics   -    numpy array
    """

    _check_n_simpl_complex(s_complex)

    if isinstance(laplacian, (tuple, list)):
        if len(laplacian) == 1:
            laplacian_res = _check_integer_values(laplacian=laplacian[0])
            L = laplace_matrix(s_complex, laplacian_res)

        elif len(laplacian) == 2:
            laplacian_res = [0, 0]

            for i in range(2):
                laplacian_res[i] = _check_integer_values(laplacian_i=laplacian[i])
            L = laplace_matrix(s_complex, laplacian_res[0], laplacian_res[1])

        elif len(laplacian) == 3:
            laplacian_res = [0, 0, 0]

            for i in range(3):
                laplacian_res[i] = _check_integer_values(laplacian_i=laplacian[i])
            L = laplace_matrix(s_complex, laplacian_res[0], laplacian_res[1], laplacian_res[2])

        else:
            raise ValueError("laplacian must be an integer or a tuple of size from 1 to 3, \
                             as it is a laplace matrix characteristic")
    else:
        laplacian_res = _check_integer_values(laplacian=laplacian)
        L = laplace_matrix(s_complex, laplacian_res)

    vals, vecs = np.linalg.eigh(L)

    cycles = vecs.T[np.abs(vals) < threshold]

    return cycles


def complex_dist(s_complex1, s_complex2, laplacian, threshold=0.01):
    """
        Функция расстояний между графами в метрике, определяющей похожесть графов
        в плане локализации их гармонических циклов (то есть определяющей схожесть
        графов с точки зрения схожести их топологии в смысле локализации их гармонических циклов)

        Parameters
        ----------
        s_complex1:         list
            Simplicial complex (list of numpy arrays)
        s_complex2:         list
            Simplicial complex (list of numpy arrays)
        laplacian:          int or tuple (k, p, q)
            Type of (k,p,q)-Laplacian
        threshold:          float
            Threshold for zero eigenvalues

        Returns
        -------
        Distance   -    float
    """

    if len(s_complex1[0]) != len(s_complex2[0]):
        raise ValueError("Simplicial complexes must have the same number of vertices")

    cycles1 = homology_cycles(s_complex1, laplacian, threshold)
    cycles2 = homology_cycles(s_complex2, laplacian, threshold)

    if len(cycles1) == 0 or len(cycles2) == 0 or len(cycles1[0]) != len(cycles2[0]):
        return -1

    res = 0
    for c1 in cycles1:
        for c2 in cycles2:
            res += _cos_sim(c1, c2)

    return res


def spectra(complexes_list, laplacian, weights_list=None):
    """
        Получение признакового описания набора графов по
        собственным значениям (k,p,q)-Лапласианов

        Parameters
        ----------
        complexes_list:     list
            List of simplicial complexes (each is a list of numpy arrays)
        laplacian:          int or tuple (k, p, q)
            Type of (k,p,q)-Laplacian
        weights_list:       list
            List of weights of simplices of each simplicial complex

        Returns
        -------
        Matrix of Laplacian eigenvalues   -    numpy array
    """

    if not isinstance(complexes_list, list):
        raise TypeError('complexes_list must be a list of complexes')

    spectra_res = []

    if weights_list is not None:
        if not isinstance(weights_list, list):
            raise TypeError('weights_list must be a list of weights')
        if len(complexes_list) != len(weights_list):
            raise ValueError('complexes_list and weights_list must be the same size')

        if isinstance(laplacian, (tuple, list)):
            if len(laplacian) == 1:
                laplacian_res = _check_integer_values(laplacian=laplacian[0])

                for c, w in tqdm(zip(complexes_list, weights_list)):
                    L = weighted_laplace_matrix(c, w, laplacian_res)
                    eigen_values_L, _ = np.linalg.eigh(L)
                    spectra_res.append(eigen_values_L)

            elif len(laplacian) == 2:
                laplacian_res = [0, 0]

                for i in range(2):
                    laplacian_res[i] = _check_integer_values(laplacian_i=laplacian[i])

                for c, w in tqdm(zip(complexes_list, weights_list)):
                    L = weighted_laplace_matrix(c, w, laplacian_res[0], laplacian_res[1])
                    eigen_values_L, _ = np.linalg.eigh(L)
                    spectra_res.append(eigen_values_L)

            elif len(laplacian) == 3:
                laplacian_res = [0, 0, 0]

                for i in range(3):
                    laplacian_res[i] = _check_integer_values(laplacian_i=laplacian[i])

                for c, w in tqdm(zip(complexes_list, weights_list)):
                    L = weighted_laplace_matrix(c, w, laplacian_res[0], laplacian_res[1], laplacian_res[2])
                    eigen_values_L, _ = np.linalg.eigh(L)
                    spectra_res.append(eigen_values_L)

            else:
                raise ValueError("laplacian must be an integer or a tuple of size from 1 to 3, \
                                 as it is a laplace matrix characteristic")
        else:
            laplacian_res = _check_integer_values(laplacian=laplacian)

            for c, w in tqdm(zip(complexes_list, weights_list)):
                L = weighted_laplace_matrix(c, w, laplacian_res)
                eigen_values_L, _ = np.linalg.eigh(L)
                spectra_res.append(eigen_values_L)

    else:
        if isinstance(laplacian, (tuple, list)):
            if len(laplacian) == 1:
                laplacian_res = _check_integer_values(laplacian=laplacian[0])

                for c in tqdm(complexes_list):
                    L = laplace_matrix(c, laplacian_res)
                    eigen_values_L, _ = np.linalg.eigh(L)
                    spectra_res.append(eigen_values_L)

            elif len(laplacian) == 2:
                laplacian_res = [0, 0]

                for i in range(2):
                    laplacian_res[i] = _check_integer_values(laplacian_i=laplacian[i])

                for c in tqdm(complexes_list):
                    L = laplace_matrix(c, laplacian_res[0], laplacian_res[1])
                    eigen_values_L, _ = np.linalg.eigh(L)
                    spectra_res.append(eigen_values_L)

            elif len(laplacian) == 3:
                laplacian_res = [0, 0, 0]

                for i in range(3):
                    laplacian_res[i] = _check_integer_values(laplacian_i=laplacian[i])

                for c in tqdm(complexes_list):
                    L = laplace_matrix(c, laplacian_res[0], laplacian_res[1], laplacian_res[2])
                    eigen_values_L, _ = np.linalg.eigh(L)
                    spectra_res.append(eigen_values_L)

            else:
                raise ValueError("laplacian must be an integer or a tuple of size from 1 to 3, \
                                 as it is a laplace matrix characteristic")
        else:
            laplacian_res = _check_integer_values(laplacian=laplacian)

            for c in tqdm(complexes_list):
                L = laplace_matrix(c, laplacian_res)
                eigen_values_L, _ = np.linalg.eigh(L)
                spectra_res.append(eigen_values_L)

    shapes = np.zeros(len(spectra_res))
    for i, s in enumerate(spectra_res):
        shapes[i] = s.shape[0]
    min_shape_L = int(shapes.min())

    for i in range(len(spectra_res)):
        spectra_res[i] = spectra_res[i][:min_shape_L]

    return np.array(spectra_res)


def get_score(model, spectra, y, n_repeats, scoring='accuracy', random_state=0):
    results = []
    for r in range(n_repeats):
        rus = RandomUnderSampler(random_state=random_state)
        spectra_res, y_res = rus.fit_resample(spectra, y)

        cv_val = list(cross_val_score(model, spectra_res, y_res, cv=5, scoring=scoring))
        results.append(cv_val)

    res = np.mean(np.array(results)) * 100
    return res
