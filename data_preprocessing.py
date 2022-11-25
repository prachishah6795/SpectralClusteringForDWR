import gurobipy as gp
import networkx as nx
import itertools
import numpy as np
import warnings
from scipy.linalg import fractional_matrix_power
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

problem_set = ['enlight_hard.mps', 'neos859080.mps']

miplib_dir = "MIPLIB 2017/"

laplacian_types = ['default', 'symmetric', 'random-walk']
graph_types = ['row-net', 'row-col-net']


def get_adjacency_matrix(G):
    # make sure that both scipy and networkx modules are updated to latest version
    return nx.adjacency_matrix(G)


def get_degree_matrix(G):
    return np.diag([d for _, d in G.degree])


def get_eigen_features(mat):
    # get first 'K' smallest eigen values and corresponding eigen vectors
    eigenvalues, eigenvectors = np.linalg.eig(mat)

    sorted_ids = eigenvalues.argsort()
    eigenvectors = eigenvectors[sorted_ids]
    eigenvalues = eigenvalues[sorted_ids]

    return eigenvalues, eigenvectors


class Instance:

    def __init__(self, name):
        model = gp.read(f"{miplib_dir}{name}")
        self.A = model.getA().todense()  # Query the linear constraint matrix of the model
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

        for v1, v2 in itertools.combinations(nodes, 2):
            if np.any(np.multiply(self.A[:, v1], self.A[:, v2]) != 0):
                self.G_row_net.add_edge(v1, v2)

        if len(list(nx.isolates(self.G_row_net))) > 0:
            warnings.warn("Graph contains isolated nodes")

    def generate_row_col_net_graph(self):
        # Each non-zero coefficient a_ij of matrix A defines a vertex vij. Two nodes are connected if
        # they belong to the same constraint or the same variable
        non0_rows, non0_cols = np.nonzero(self.A)
        nodes = list(zip(non0_rows, non0_cols))
        self.G_row_col_net.add_nodes_from(nodes)

        for v1, v2 in itertools.combinations(nodes, 2):
            if v1[0] == v2[0] or v1[1] == v2[1]:
                self.G_row_col_net.add_edge(v1, v2)

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
        return kmeans

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
            self.L[graph_type][laplacian_types[1]] = fractional_matrix_power(D, -0.5) @ L \
                                                     @ fractional_matrix_power(D, 0.5)  # symmetric
            self.L[graph_type][laplacian_types[2]] = np.linalg.inv(D) @ L  # r-w


out = {}
for problem in problem_set:
    instance = Instance(problem)
    out[problem] = instance.spectral_clustering(graph_types[0], laplacian_types[0], K=5)
