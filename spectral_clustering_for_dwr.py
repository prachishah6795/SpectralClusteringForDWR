import os
import pickle
import gurobipy as gp
import networkx as nx
import itertools
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.linalg import fractional_matrix_power
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import matplotlib.pylab as plt
import matplotlib.patches as patches

miplib_dir = "MIPLIB Instances/"
problem_set = os.listdir(miplib_dir)

laplacian_types = ['default', 'symmetric', 'random-walk']
graph_types = ['row-net', 'row-col-net']


def get_adjacency_matrix(G):
    # make sure that both scipy and networkx modules are updated to latest version
    return nx.adjacency_matrix(G).toarray()


def get_degree_matrix(G):
    return np.diag([d for _, d in G.degree(weight='weight')])


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def get_eigen_features(mat):
    # get first 'K' smallest eigen values and corresponding eigen vectors
    eigenvalues, eigenvectors = np.linalg.eigh(mat)

    return eigenvalues, eigenvectors


class Instance:

    def __init__(self, name):
        model = gp.read(f"{miplib_dir}{name}")
        self.A = model.getA().toarray()  # Query the linear constraint matrix of the model
        self.m, self.n = self.A.shape  # m = number of constraints; n = number of variables
        self.G_row_net = nx.Graph()
        self.G_row_col_net = nx.Graph()
        self.generate_row_net_graph()
        self.generate_row_col_net_graph()
        # self.W = None  # adjacency matrix
        # self.D = None  # degree_matrix
        self.L = {}

    def generate_row_net_graph(self):
        # Each column (variable) j of matrix A defines a vertex vj.
        # Two nodes are connected if there exists a constraint where both of these variables have non-zero coefficients
        nodes = range(self.n)
        self.G_row_net.add_nodes_from(nodes)
        count_non_zeros = np.count_nonzero(self.A, axis=1)
        # count_non_zeros[count_non_zeros > 0.6*self.n] = 0

        for v1, v2 in itertools.combinations(nodes, 2):
            weight = np.sum(np.where(np.multiply(self.A[:, v1], self.A[:, v2])!=0, 1/count_non_zeros**2, 0))
            if weight > 0:
                self.G_row_net.add_edge(v1, v2, weight=weight)

        isolated_nodes = list(nx.isolates(self.G_row_net))
        if len(isolated_nodes) > 0:
            print(f"Graph contains {len(isolated_nodes)} isolated nodes. Removing them.")
            self.A = np.delete(self.A, isolated_nodes, 1)
            self.n -= 1
            self.G_row_net.remove_nodes_from(list(nx.isolates(self.G_row_net)))

    def generate_row_col_net_graph(self):
        # Each non-zero coefficient a_ij of matrix A defines a vertex vij. Two nodes are connected if
        # they belong to the same constraint or the same variable
        non0_rows, non0_cols = np.nonzero(self.A)
        nodes = list(zip(non0_rows, non0_cols))
        self.G_row_col_net.add_nodes_from(nodes)

        count_non_zeros_rows = np.count_nonzero(self.A, axis=1)
        count_non_zeros_cols = np.count_nonzero(self.A, axis=0)

        for v1, v2 in itertools.combinations(nodes, 2):
            if v1[0] == v2[0]:
                self.G_row_col_net.add_edge(v1, v2, weight=1/count_non_zeros_rows[v1[0]]**2)
            if v1[1] == v2[1]:
                self.G_row_col_net.add_edge(v1, v2, weight=1/count_non_zeros_cols[v1[1]]**2)

        if len(list(nx.isolates(self.G_row_col_net))) > 0:
            print(f"Graph contains {len(list(nx.isolates(self.G_row_col_net)))} isolated nodes")

    def spectral_clustering(self, graph_type, laplacian_type, K=5):
        assert graph_type in graph_types, 'check input'
        assert laplacian_type in laplacian_types, 'check input'

        if graph_type not in self.L:
            self.gen_graph_laplacians(graph_type)

        L = self.L[graph_type][laplacian_type]

        w, v = get_eigen_features(L)
        U = v[:, 0:K].real

        if laplacian_type == laplacian_types[1]:
            U = normalize(U, norm='l2', axis=1)

        kmeans = KMeans(n_clusters=K).fit(U)

        A_dwr, row_sizes, col_sizes = self.get_rearranged_matrix(kmeans.labels_, graph_type)

        return A_dwr, row_sizes, col_sizes

    def gen_graph_laplacians(self, graph_type='row-net', overwrite=False):
        # graph type in ['row-net', 'row-col-net']
        # laplacian type in ['default', 'symmetric', 'random-walk']
        # https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf

        G = self.G_row_net if graph_type == graph_types[0] else self.G_row_col_net
        W = get_adjacency_matrix(G)
        D = get_degree_matrix(G)

        L = D - W
        if graph_type not in self.L or overwrite:
            self.L[graph_type] = {}
            self.L[graph_type][laplacian_types[0]] = L  # default
            self.L[graph_type][laplacian_types[1]] = np.identity(D.shape[0]) - \
                                                     fractional_matrix_power(D, -0.5) @ W @ fractional_matrix_power(D, -0.5)  # symmetric
            self.L[graph_type][laplacian_types[2]] = np.identity(D.shape[0]) - np.linalg.inv(D) @ W  # r-w

    def get_rearranged_matrix(self, labels, graph_type='row-net'):
        assert graph_type in graph_types, 'check input'

        A_dwr = self.A.copy()
        if graph_type == graph_types[0]:

            labelled_nodes = np.array(sorted(zip(labels, range(self.n))))
            A_dwr = A_dwr[:, labelled_nodes[:, 1]]  # rearranging columns by clusters
            A_abs_sum_rows = np.sum(np.abs(self.A), axis=1).flatten()

            # Rearrange rows
            unique_labels, unique_indices = np.unique(labelled_nodes[:, 0], return_index=True)
            var_clusters = np.split(labelled_nodes[:, 1], unique_indices[1:])

            row_cluster = []
            for i in range(len(unique_labels)):
                vars_in_cluster = var_clusters[i]
                rows_in_cluster = np.argwhere(np.sum(np.abs(self.A[:, vars_in_cluster]), axis=1).flatten() == A_abs_sum_rows)
                row_cluster.append(rows_in_cluster.flatten())

            col_block_sizes = []
            for each in var_clusters:
                col_block_sizes.append(each.size)

            row_block_sizes = []
            row_order = []
            for each in row_cluster:
                row_block_sizes.append(each.size)
                row_order.extend(each)

            linking_cons = set(range(self.m)) - set(row_order)
            row_order.extend(linking_cons)
            row_block_sizes.append(len(linking_cons))

            A_dwr = A_dwr[row_order, :]

        if graph_type == graph_types[1]:
            labelled_nodes = np.array(sorted(zip(labels, self.G_row_col_net.nodes)), dtype=object)
            unique_labels, unique_indices = np.unique(labelled_nodes[:, 0], return_index=True)
            coef_clusters = np.split(labelled_nodes[:, 1], unique_indices[1:])

            cons_clusters, var_clusters = [], []
            for cluster in coef_clusters:
                cons, vars = zip(*cluster)
                cons_clusters.append(np.unique(np.fromiter(cons, dtype=int)))
                var_clusters.append(np.unique(np.fromiter(vars, dtype=int)))

            linking_vars = set(v for v in range(self.n) if sum([v in cl for cl in var_clusters]) >= 2)
            linking_cons = set(c for c in range(self.m) if sum([c in cl for cl in cons_clusters]) >= 2)

            col_order = []
            col_block_sizes = []
            for cl in var_clusters:
                cluster = set(cl) - linking_vars
                col_order.extend(cluster)
                col_block_sizes.append(len(cluster))
            col_order.extend(linking_vars)
            col_block_sizes.append(len(linking_vars))

            row_order = []
            row_block_sizes = []
            for cl in cons_clusters:
                cluster = set(cl) - linking_cons
                row_order.extend(cluster)
                row_block_sizes.append(len(cluster))
            row_order.extend(linking_cons)
            row_block_sizes.append(len(linking_cons))

            A_dwr = A_dwr[row_order, :]
            A_dwr = A_dwr[:, col_order]

        return A_dwr, np.array(row_block_sizes), np.array(col_block_sizes)


if __name__ == "__main__":
    miplib_dir = "MIPLIB Instances/"
    problem_set = os.listdir(miplib_dir)

    laplacian_types = ['default', 'symmetric', 'random-walk']
    graph_types = ['row-net', 'row-col-net']

    out = []
    results = []
    pred_k = []
    # output = {}
    for problem in problem_set:

        instance = Instance(problem)
        problem = problem.replace('.mps.gz', '')
        Path(f"Images/{problem}/").mkdir(exist_ok=True, parents=True)

        init_matrix = instance.A
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 6)
        ax.spy(init_matrix, markersize=1, mec='black', aspect='auto')
        plt.savefig(f"Images/{problem}/{problem}.png")
        plt.close()

        for graph in graph_types:
            instance.gen_graph_laplacians(graph)

            for laplacian in laplacian_types:
                L = instance.L[graph][laplacian]
                w, v = get_eigen_features(L)

                w_diff = w[2:11] - w[1:10]
                k_hat = 2 + np.where(w_diff == w_diff.max())[0]

                for k in range(2, 11):
                    print(problem, laplacian, graph, k)

                    dwr_matrix, row_sizes, col_sizes = instance.spectral_clustering(graph, laplacian, K=k)
                    # output[(problem, laplacian, graph, k)] = (dwr_matrix, row_sizes, col_sizes)

                    row_sep = np.cumsum(row_sizes)
                    col_sep = np.cumsum(col_sizes)

                    assert row_sep[-1] == instance.m, "row sep compute error"
                    assert col_sep[-1] == instance.n, "col sep compute error"

                    m_l = row_sizes[-1]
                    n_l = col_sizes[-1] if graph == graph_types[1] else 0

                    pct_linking_cons = m_l / instance.m
                    pct_linking_vars = n_l / instance.n

                    border_area = (m_l * instance.n + instance.m * n_l - m_l * n_l) / (instance.m * instance.n)
                    block_areas = row_sizes[:k] * col_sizes[:k]
                    block_area_diff = np.max(block_areas) / np.min(block_areas)

                    results.append(
                        [problem, laplacian, graph, k, pct_linking_cons, pct_linking_vars,
                         border_area, block_area_diff])

                    if k in k_hat:
                        pred_k.append(
                            [problem, laplacian, graph, k, pct_linking_cons, pct_linking_vars,
                             border_area, block_area_diff])

                    # For plotting
                    fig, ax = plt.subplots()
                    fig.set_size_inches(10, 6)
                    ax.spy(dwr_matrix, markersize=1, mec='black', aspect='auto')
                    xy = (0, 0)
                    for x in range(k):
                        rect = patches.Rectangle(xy, col_sizes[x], row_sizes[x], linewidth=1, edgecolor='blue',
                                                 facecolor='powderblue')
                        ax.add_patch(rect)
                        xy = (col_sep[x], row_sep[x])

                    # Linking constraints
                    rect = patches.Rectangle((0, row_sep[-2]), instance.n, row_sizes[-1], linewidth=1, edgecolor='blue',
                                             facecolor='powderblue')
                    ax.add_patch(rect)

                    if graph == graph_types[1]:
                        # Linking variables
                        rect = patches.Rectangle((col_sep[-2], 0), col_sizes[-1], instance.m, linewidth=1,
                                                 edgecolor='blue', facecolor='powderblue')
                        ax.add_patch(rect)

                    # plt.show()
                    plt.savefig(f"Images/{problem}/{problem}_{laplacian}_{graph}_{k}.png")
                    plt.close()

    # with open("dwr_output_2.pickle", "wb") as handle:
    #     pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)


    results_df = pd.DataFrame(results,
                              columns=['Problem',
                                       'Laplacian',
                                       'Graph',
                                       'K',
                                       '% Linking Constraints',
                                       '% Linking Variables',
                                       'Relative Border Area',
                                       'Block Size Ratio'])

    predi_k_df = pd.DataFrame(pred_k,
                              columns=['Problem',
                                       'Laplacian',
                                       'Graph',
                                       'K',
                                       '% Linking Constraints',
                                       '% Linking Variables',
                                       'Relative Border Area',
                                       'Block Size Ratio'])

    with pd.ExcelWriter('dwr_results_2.xlsx', mode='w') as writer:
        results_df.to_excel(writer, sheet_name='Results')
        predi_k_df.to_excel(writer, sheet_name='PredictedK')