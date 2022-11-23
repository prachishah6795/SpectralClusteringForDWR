import gurobipy as gp
import networkx as nx
import itertools
import numpy as np
import warnings


problem_set = ['enlight_hard', 'neos859080']

miplib_dir = "MIPLIB 2017/"


class Instance:

    def __init__(self, name):
        model = gp.read(f"{miplib_dir}{name}.mps.gz")
        self.A = model.getA().todense()
        self.m, self.n = self.A.shape
        self.G_row_net = nx.Graph()
        self.G_row_col_net = nx.Graph()

    def generate_row_net_graph(self):
        # Each column (variable) j of matrix A defines a vertex vj. Two nodes are connected if there exists a constraint
        # where both of these variables have non-zero coefficients
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

    def get_eigen_features(self, graph_type='row-net', laplacian_type='unnormalized'):
        # graph type in ['row-net', 'row-col-net']
        # laplacian type in ['default', 'symmetric', 'random-walk']
        # https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf

        G = self.G_row_net if graph_type=='row-net' else self.G_row_col_net


for problem in problem_set:

    instance = Instance(problem)
    instance.generate_row_net_graph()
    instance.generate_row_col_net_graph()