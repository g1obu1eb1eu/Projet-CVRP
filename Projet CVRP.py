from random import randint
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def graph():
    dimension = randint(3, 10)
    my_graph = [[0 for _ in range(dimension)] for _ in range(dimension)]

    for i in range(dimension):
        for k in range(i + 1, dimension):
            val = randint(1, 20)
            my_graph[i][k], my_graph[k][i] = val, val

    return my_graph


def show_graph(sac):
    rows, cols = np.where(sac)
    edges = zip(list(rows), list(cols))
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500)
    plt.show()


sac = graph()
show_graph(sac)
