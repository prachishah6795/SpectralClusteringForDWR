import gurobipy as gp
import networkx as nx
import itertools
import numpy as np
import warnings
from scipy.linalg import fractional_matrix_power
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import matplotlib.pylab as plt

problem_set = ['rout.mps.gz', 'fiber.mps.gz', 'timtab1.mps.gz']# , '10teams.mps.gz']

miplib_dir = "MIPLIB 2003/"

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

    # sorted_ids = eigenvalues.argsort()
    # eigenvectors = eigenvectors[sorted_ids]
    # eigenvalues = eigenvalues[sorted_ids]

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
            weight = np.sum(np.where(np.multiply(self.A[:, v1], self.A[:, v2])!=0, 1/count_non_zeros, 0))
            if weight > 0:
                self.G_row_net.add_edge(v1, v2, weight=weight)

        if len(list(nx.isolates(self.G_row_net))) > 0:
            warnings.warn("Graph contains isolated nodes")

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
                self.G_row_col_net.add_edge(v1, v2, weight=1/count_non_zeros_rows[v1[0]])
            if v1[1] == v2[1]:
                self.G_row_col_net.add_edge(v1, v2, weight=1/count_non_zeros_cols[v1[1]])

        if len(list(nx.isolates(self.G_row_col_net))) > 0:
            warnings.warn("Graph contains isolated nodes")

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
        print(kmeans.n_iter_)

        A_dwr = self.get_rearranged_matrix(kmeans.labels_, graph_type)

        return A_dwr

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

            print("var cluster sizes")
            for each in var_clusters:
                print(each.shape, end=', ')
            print()

            print("cons cluster sizes")
            row_order = []
            for each in row_cluster:
                print(each.shape, end=', ')
                row_order.extend(each)

            linking_rows = set(range(self.m)) - set(row_order)
            row_order.extend(linking_rows)
            print(len(linking_rows))

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
            for cl in var_clusters:
                col_order.extend(set(cl) - linking_vars)
                print(len(set(cl) - linking_vars), end=', ')
            col_order.extend(linking_vars)
            print(len(linking_vars))

            row_order = []
            for cl in cons_clusters:
                row_order.extend(set(cl) - linking_cons)
                print(len(set(cl) - linking_cons), end=', ')
            row_order.extend(linking_cons)
            print(len(linking_cons))

            A_dwr = A_dwr[row_order, :]
            A_dwr = A_dwr[:, col_order]

        return A_dwr



out = {}
for problem, k in zip(problem_set, [4, 3, 2]):
    instance = Instance(problem)
    print(f"matrix size = {instance.m}, {instance.n}")
    init_matrix = instance.A
    out[problem] = instance.spectral_clustering(graph_types[1], laplacian_types[1], K=k)
    rearranged_matrix = out[problem]

    plt.spy(init_matrix, markersize=1)
    plt.show()

    plt.spy(rearranged_matrix, markersize=1)
    plt.show()


