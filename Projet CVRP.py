from random import randint
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import vrplib
from pathlib import Path

# Path to CVRP instance directory
path_to_instances = Path(__file__).parent / "full_dataset"

# Path to specific instance & solution files
instance_file = path_to_instances / "X-n101-k25.vrp"
solution_file = path_to_instances / "X-n101-k25.sol"

if not instance_file:
    raise FileNotFoundError(f"Instance file not found at : {instance_file}")

if not solution_file:
    raise FileNotFoundError(f"Solution file not found at : {solution_file}")

# Read VRPLIB formatted instances (default)
instance = vrplib.read_instance(instance_file)
solution = vrplib.read_solution(solution_file)

# instance_items = instance.items()
# solution_items = solution.items()

# for key, val in instance_items:
#     print(f"{key} : {val}\n")
# print("---------------------")
# for key, val in solution_items:
#     print(f"{key} : {val}\n")

# print(f"{instance["edge_weight"]}")

x_coord = [x for x, y in instance["node_coord"]]
y_coord = [y for x, y in instance["node_coord"]]

plt.scatter(x_coord, y_coord, edgecolors="red")
plt.plot(x_coord, y_coord, "-o")

plt.show()

# def graph():
#     dimension = randint(3, 10)
#     my_graph = [[0 for _ in range(dimension)] for _ in range(dimension)]

#     for i in range(dimension):
#         for k in range(i + 1, dimension):
#             val = randint(1, 20)
#             my_graph[i][k], my_graph[k][i] = val, val

#     return my_graph


def show_graph(sac):
    rows, cols = np.where(sac)
    edges = zip(list(rows), list(cols))
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500)
    plt.show()


# sac = graph()
show_graph(instance["edge_weight"])
