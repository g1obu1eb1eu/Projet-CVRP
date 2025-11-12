from random import randint
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import vrplib
from pathlib import Path
import pulp
import time

# Path to CVRP instance directory
path_to_instances = Path(__file__).parent / "full_dataset"

# Path to specific instance & solution files

# Dataset 1
instance_file = path_to_instances / "H-n32-k6.vrp"
solution_file = path_to_instances / "H-n32-k6.sol"

# # Dataset 2
# instance_file = path_to_instances / "X-n101-k25.vrp"
# solution_file = path_to_instances / "X-n101-k25.sol"


if not instance_file:
    raise FileNotFoundError(f"Instance file not found at : {instance_file}")

if not solution_file:
    raise FileNotFoundError(f"Solution file not found at : {solution_file}")

# Read VRPLIB formatted instances (default)
instance = vrplib.read_instance(instance_file)
solution = vrplib.read_solution(solution_file)


# Instances initialisation
coords = instance["node_coord"]
demands = instance["demand"]
capacity = instance["capacity"]

nb_nodes = instance["dimension"]
nodes = list(range(1, nb_nodes + 1))
depot = instance["depot"]
customers = [i for i in nodes if i != depot]

nb_trucks = None
comment = instance["comment"]

if comment:
    import re
    num_trucks = re.search(r"No of trucks[:\s]*(\d+)\s*,", comment)

    if num_trucks:
        nb_trucks = int(num_trucks.group(1))

if nb_trucks is None:
    print("eroooooooooor")

# instance_items = instance.items()
# solution_items = solution.items()

# for key, val in instance_items:
#     print(f"{key} : {val}\n")
# print("---------------------")
# for key, val in solution_items:
#     print(f"{key} : {val}\n")

# print(f"{instance['edge_weight']}")


# def graph():
#     dimension = randint(3, 10)
#     my_graph = [[0 for _ in range(dimension)] for _ in range(dimension)]

#     for i in range(dimension):
#         for k in range(i + 1, dimension):
#             val = randint(1, 20)
#             my_graph[i][k], my_graph[k][i] = val, val

#     return my_graph


# def show_graph(edge_weight, node_coord):
#     gr = nx.Graph()
#     num_nodes = len(node_coord)
#     edges = []
#     for i in range(num_nodes):
#         for j in range(num_nodes):
#             if i != j:
#                 w = float(edge_weight[i, j])
#                 if w > 0:
#                     edges.append((i, j, {"weight": w}))

#     gr.add_edges_from(edges)

#     # pos = {point: point for point in node_coord}

#     pos = {
#         i: (float(node_coord[i][0]), float(node_coord[i][1]))
#         for i in range(len(node_coord))
#     }
#     # print(pos)

#     fig, ax = plt.subplots()
#     nx.draw(gr, pos=pos, node_size=500, node_color="skyblue", edge_color="black", ax=ax)
#     # nx.draw_networkx_labels(gr, pos=pos)
#     # labels = nx.get_edge_attributes(gr, "weight")
#     # nx.draw_networkx_edge_labels(gr, pos, edge_labels=labels, ax=ax)
#     plt.axis("on")
#     ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
#     plt.show()


# show_graph(instance["edge_weight"], instance["node_coord"])


def linear_program(node_coord):
    n = len(node_coord)

    # Initialisation resolution pblm hrcvrpcc
    prob = pulp.LpProblem("HRCVRPP", pulp.LpMinimize)

    x = {
        (i, j): pulp.LpVariable(f"x{{{i}_{j}}}", cat="Binary")
        for i in range(n)
        for j in range(n)
        if (i != j)
    }
