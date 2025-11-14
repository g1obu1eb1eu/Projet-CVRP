# --------------------------- Importation des bibliothèques --------------------------

import matplotlib.pyplot as plt
import vrplib
from pathlib import Path
import networkx as nx
import numpy as np
import math
import time
import random
import pulp
from collections import Counter

# --------------------------- Initialisation des Instances ---------------------------

# Path to CVRP instance directory
path_to_instances = (
    Path(__file__).parent / "full_dataset"
)  # dossier contenant les instances.

# Path to specific instance & solution files

# Dataset 1
# instance_file = path_to_instances / "H-n32-k6.vrp"
# solution_file = path_to_instances / "H-n32-k6.sol"  # solution associée

# Dataset 2
# instance_file = path_to_instances / "H-n100-k27.vrp"
# solution_file = path_to_instances / "H-n100-k27.sol"

# Dataset 3
instance_file = path_to_instances / "H-n200-k18.vrp"
solution_file = path_to_instances / "H-n200-k18.sol"


if not instance_file:
    raise FileNotFoundError(
        f"Instance file not found at : {instance_file}"
    )  # vérifie le chemin

if not solution_file:
    raise FileNotFoundError(f"Solution file not found at : {solution_file}")

# Read VRPLIB formatted instances
instance = vrplib.read_instance(instance_file)
solution = vrplib.read_solution(solution_file)


# Instances initialisation
coords = instance["node_coord"]  # coordonnées (x,y) des noeuds
demands = instance["demand"]  # demandes des clients
capacity = instance["capacity"]  # capacité des camions

nb_nodes = instance["dimension"]  # nombre de noeuds total
nodes = list(range(1, nb_nodes + 1))  # liste 1..nb_nodes

depot = int(instance["depot"].item() + 1)  # index depot (1)
customers = [i for i in nodes if i != depot]  # liste des clients (exclu dépôt)

cost = int(solution["cost"])

# print(f"{depot}")  # debug: afficher depot

nb_trucks = None
comment = instance["comment"]  # champ comment du fichier VRP

# print(nodes)

if comment:
    import re

    num_trucks = re.search(r"No of trucks[:\s]*(\d+)\s*\,", comment)

    if num_trucks:
        nb_trucks = int(num_trucks.group(1))  # extraire nombre de camions

if nb_trucks is None:
    # print("erroooooooooor")
    raise ValueError("val not found")  # stop si non trouvé

K = list(range(nb_trucks))

edge_weight = instance["edge_weight"]  # matrice des distances/poids

# print(len(edge_weight))  # debug: afficher taille matrice

c = {
    (i, j): (0 if i - 1 == j - 1 else edge_weight[i - 1, j - 1])
    for i in nodes
    for j in nodes
}  # coût entre i et j, diag=0

d = {i: int(demands[i - 1]) for i in nodes}  # demande par noeud
# d[depot] = 0

Q = {k: capacity for k in range(1, nb_trucks + 1)}  # capacité par camion

truck_type = {
    k: int(instance["truck_type"][k - 1]) for k in range(1, nb_trucks + 1)
}  # type camion

cust_type = {
    i: int(instance["vehicle_type"][i - 1]) for i in customers
}  # type requis par client

compat = {
    (i, k): int(truck_type[k] == cust_type[i])
    for i in customers
    for k in range(1, nb_trucks + 1)
}  # compatibilité binaire client-camion


# print(compat)

# --------------------------------- Progamme Linéaire --------------------------------


def linear_program(nb_trucks, nodes, compat):
    F = {k: 0.0 for k in range(1, nb_trucks + 1)}  # coût fixe véhicule (ici 0)

    prob = pulp.LpProblem("HCVRPCC", pulp.LpMinimize)  # modèle pulp (minimisation)

    K = list(range(1, nb_trucks + 1))
    V = nodes
    x = pulp.LpVariable.dicts(
        "x", (K, V, V), cat="Binary"
    )  # x[k][i][j] = 1 si arc i->j utilisé par k
    y = pulp.LpVariable.dicts("y", K, cat="Binary")  # y[k] = 1 si véhicule k utilisé
    f = pulp.LpVariable.dicts(
        "f", (K, V, V), lowBound=0, cat="Continuous"
    )  # flux (pour capacités)

    # Fonction obj
    prob += pulp.lpSum(
        c[i, j] * x[k][i][j] for k in K for i in V for j in V if i != j
    ) + pulp.lpSum(F[k] * y[k] for k in K)

    # Chaque cleint visité une fois
    for j in customers:
        prob += (
            pulp.lpSum(x[k][i][j] for k in K for i in V if i != j) == 1,
            f"visit_once_{j}",
        )  # contrainte: chaque client servi exactement 1 fois

    # entrée == sortie, pour chaque véhicule en chaque noeuds
    for k in K:
        for i in V:
            prob += (
                pulp.lpSum(x[k][i][j] for j in V if j != i)
                == pulp.lpSum(x[k][j][i] for j in V if j != i),
                f"flow_consistency_k{k}_i{i}",
            )  # conservation de flot

    # départ et retour au dépot
    for k in K:
        for i in customers:
            prob += (
                pulp.lpSum(x[k][i][j] for j in V if j != i) <= compat[(i, k)],
                f"dep_k{k}_i{i}",
            )  # si non compatible, on ne peut pas partir de i avec k

    # activation arcs selon y_k
    for k in K:
        for i in V:
            for j in V:
                if i != j:
                    prob += x[k][i][j] <= y[k], f"activate_k : {k}_i{i}_j{j}"

    # compatibilité truncks-client
    for k in K:
        for i in customers:
            prob += (
                pulp.lpSum(x[k][i][j] for j in V if j != i) <= compat[(i, k)],
                f"compat_k{k}_i{i}",
            )  # double contrainte compatibilité

    # capacité -> la somme des demandes desservies doit être <= Q_k * y_k
    for k in K:
        prob += (
            pulp.lpSum(
                d[i] * pulp.lpSum(x[k][i][j] for j in V if j != i) for i in customers
            )
            <= Q[k] * y[k],
            f"capacity_k{k}",
        )  # capacité totale par véhicule

    # bornes flux 0 <= f <= Q_k * x
    for k in K:
        for i in V:
            for j in V:
                if i != j:
                    prob += f[k][i][j] <= Q[k] * x[k][i][j], f"flow_ub_k{k}_i{i}_j{j}"
                    prob += f[k][i][j] >= 0, f"flow_lb_k{k}_i{i}_j{j}"

    # conservation charge pour les clients
    for k in K:
        for i in customers:
            prob += (
                pulp.lpSum(f[k][j][i] for j in V if j != i)
                - pulp.lpSum(f[k][i][j] for j in V if j != i)
                == d[i] * pulp.lpSum(x[k][i][j] for j in V if j != i),
                f"charge_cons_k{k}_i{i}",
            )  # équilibre flux=demande si client desservi

    # conservation flux au depot
    for k in K:
        prob += (
            pulp.lpSum(f[k][depot][j] for j in V if j != depot)
            == pulp.lpSum(
                d[i] * pulp.lpSum(x[k][i][j] for j in V if j != i) for i in customers
            ),
            f"depot_flow_k{k}",
        )  # total sortant du dépôt = total desservi

    return prob, K, V, x


# Résolution du pblm via pulp
def fonction_solver(prob):
    time_limit = 15
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)  # utilise CBC
    prob.solve(solver)  # solve


def show_roads(K, V, x):
    all_routes = []
    for k in K:
        route = [depot]
        current = depot
        visited = set()

        while True:
            next_node = None
            for j in V:
                if j != current and j != depot and pulp.value(x[k][current][j]) > 0.5:
                    next_node = j
                    break

            if next_node is None:
                break

            route.append(next_node)
            visited.add(next_node)
            current = next_node

        if len(route) > 1:
            route.append(depot)
            all_routes.append(route)
            print(f"Truck {k}: {route}")

    return all_routes


# print(f"File name: {instance_file.name}")
# print(
#     f"Dimension: {nb_nodes}, depot node num: {depot}, customers: {len(customers)}, vehicles (nb_trucks): {nb_trucks}, capacity: {capacity}, cost_opti: {cost}"
# )


# def show_file_content():
#     instance_items = instance.items()
#     solution_items = solution.items()

#     for key, val in instance_items:
#         print(f"{key} : {val}\n")
#     print("---------------------")
#     for key, val in solution_items:
#         print(f"{key} : {val}\n")

#     print(f"{instance['edge_weight']}")


# show_file_content()

# -------- Initialisation de la solution initiale - pour les meta-heuristiques -------


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])  # distance euclidienne


dist = [[0.0] * nb_nodes for _ in range(nb_nodes)]
for i in range(nb_nodes):
    for j in range(nb_nodes):
        dist[i][j] = euclidean(coords[i], coords[j])  # remplir matrice dist


def route_cost(route):
    if not route:
        return 0.0
    # le dépôt est le nœud 1 (index 0 dans dist)
    cost = dist[0][route[0] - 1]  # depot -> premier client
    for a, b in zip(route, route[1:]):
        cost += dist[a - 1][b - 1]
    cost += dist[route[-1] - 1][0]  # dernier client -> depot
    return cost


def total_cost(solution):
    return sum(route_cost(r) for r in solution)  # somme des coûts


def route_load(route):
    return sum(d[i] for i in route)  # load total d'une route


def initial_solution_greedy_for_type(
    customers_subset, max_trucks, capacity_val=int(capacity)
):
    unserved = set(customers_subset)
    routes = []

    # tant qu'il reste des clients
    while unserved:
        # si on peut encore créer des routes neuves -> créer une route de départ (seed)
        if len(routes) < max_trucks:
            # seed customer : prendre celui avec plus grande demande (heuristique pour moins de routes)
            seed = max(unserved, key=lambda x: d[x])
            unserved.remove(seed)
            route = [seed]
            load = d[seed]
            changed = True
            # insérer greedy best insertion
            while changed:
                changed = False
                best_cust = None
                best_pos = None
                best_incr = float("inf")
                for cust in list(unserved):
                    if load + d[cust] > capacity_val:
                        continue
                    for pos in range(len(route) + 1):
                        new_route = route[:pos] + [cust] + route[pos:]
                        incr = route_cost(new_route) - route_cost(route)
                        if incr < best_incr:
                            best_incr = incr
                            best_cust = cust
                            best_pos = pos
                if best_cust is not None:
                    route.insert(best_pos, best_cust)
                    unserved.remove(best_cust)
                    load += d[best_cust]
                    changed = True
            routes.append(route)
        else:
            # on a atteint le nombre maximum de routes autorisées
            # on doit insérer les clients restants dans les routes existantes (si possible)
            progressed = False
            for cust in list(unserved):
                best_incr = float("inf")
                best_route_idx = None
                best_pos = None
                for ri, route in enumerate(routes):
                    if route_load(route) + d[cust] > capacity_val:
                        continue
                    for pos in range(len(route) + 1):
                        new_route = route[:pos] + [cust] + route[pos:]
                        incr = route_cost(new_route) - route_cost(route)
                        if incr < best_incr:
                            best_incr = incr
                            best_route_idx = ri
                            best_pos = pos
                if best_route_idx is not None:
                    routes[best_route_idx].insert(best_pos, cust)
                    unserved.remove(cust)
                    progressed = True
            if not progressed:
                # pas possible d'insérer les clients restants dans les routes existantes
                raise ValueError(
                    f"Infeasible: cannot serve remaining {len(unserved)} customers with {max_trucks} trucks (type group)."
                )
    return routes


# ----------------------------------- Recuit Simulé ----------------------------------


def relocate(solution):
    # choose two different routes with at least one customer in from_route
    if len(solution) < 1:
        return solution
    sol = [r[:] for r in solution]
    # pick non-empty from_route
    from_idx = random.randrange(len(sol))
    if not sol[from_idx]:
        return sol
    cust_pos = random.randrange(len(sol[from_idx]))
    cust = sol[from_idx].pop(cust_pos)
    # if route became empty, leave it and remove later if empty
    # try to insert into some route (including new route at end)
    possible_positions = []
    for to_idx in range(len(sol)):
        if to_idx == from_idx:
            continue
        if route_load(sol[to_idx]) + demands[cust - 1] <= capacity:
            for pos in range(len(sol[to_idx]) + 1):
                possible_positions.append((to_idx, pos))
    # Option to create a new route (if capacity allows)
    if demands[cust - 1] <= capacity:
        possible_positions.append((len(sol), 0))  # append as new route
    if not possible_positions:
        # revert: put customer back
        sol[from_idx].insert(cust_pos, cust)
        return sol
    to_idx, pos = random.choice(possible_positions)
    if to_idx == len(sol):
        sol.append([cust])
    else:
        sol[to_idx].insert(pos, cust)
    # remove empty routes
    sol = [r for r in sol if r]
    return sol


def swap_customers(solution):
    sol = [r[:] for r in solution]
    if sum(len(r) for r in sol) < 2 or len(sol) < 1:
        return sol
    # pick two distinct customers in possibly different routes
    # flatten index -> (route_idx, pos)
    candidates = []
    for ri, r in enumerate(sol):
        for pi in range(len(r)):
            candidates.append((ri, pi))
    (r1, p1), (r2, p2) = random.sample(candidates, 2)
    c1 = sol[r1][p1]
    c2 = sol[r2][p2]
    # check capacity after swap
    load1 = route_load(sol[r1]) - demands[c1 - 1] + demands[c2 - 1]
    load2 = route_load(sol[r2]) - demands[c2 - 1] + demands[c1 - 1]
    if load1 <= capacity and load2 <= capacity:
        sol[r1][p1], sol[r2][p2] = sol[r2][p2], sol[r1][p1]
        # remove empty routes (unlikely)
        sol = [r for r in sol if r]
        return sol
    else:
        return sol  # no change if infeasible


def two_opt_intra(solution):
    sol = [r[:] for r in solution]
    routes_with_len = [i for i, r in enumerate(sol) if len(r) >= 4]
    if not routes_with_len:
        return sol
    ri = random.choice(routes_with_len)
    r = sol[ri]
    i, k = sorted(random.sample(range(len(r)), 2))
    if i == k:
        return sol
    new_r = r[:i] + list(reversed(r[i : k + 1])) + r[k + 1 :]
    sol[ri] = new_r
    return sol


def random_neighbor(solution):
    op = random.random()
    if op < 0.45:
        return relocate(solution)
    elif op < 0.8:
        return swap_customers(solution)
    else:
        return two_opt_intra(solution)


def accept_cost(delta, T):
    if delta <= 0:
        return True
    else:
        return random.random() < math.exp(-delta / T)  # probabilité d'acceptation


def solve_type_group(type_id, truck_count):
    customers_t = [i for i in customers if cust_type[i] == type_id]
    if not customers_t:
        return [], 0.0

    # 1) solution initiale
    try:
        init_routes_t = initial_solution_greedy_for_type(customers_t, truck_count)
    except ValueError as e:
        raise ValueError(f"Infeasible allocation with {truck_count} trucks: {e}")

    init_cost_t = total_cost(init_routes_t)

    # 2) simulated annealing
    best_t, best_cost_t = simulated_annealing(
        init_routes_t,
        T0=2000.0,
        alpha=0.995,
        iter_per_T=250,
        Tmin=0.5,
        max_no_improve=1500,
    )

    return init_routes_t, init_cost_t, best_t, best_cost_t


def simulated_annealing(
    initial_solution,
    T0=1000.0,
    alpha=0.995,
    iter_per_T=200,
    Tmin=1e-3,
    max_no_improve=1000,
):
    current = initial_solution
    current_cost = total_cost(current)
    best = [r[:] for r in current]
    best_cost = current_cost
    T = T0
    no_improve = 0
    it = 0
    while T > Tmin and no_improve < max_no_improve:
        for _ in range(iter_per_T):
            neighbor = random_neighbor(current)
            neigh_cost = total_cost(neighbor)
            delta = neigh_cost - current_cost
            if accept_cost(delta, T):
                current = neighbor
                current_cost = neigh_cost
                if current_cost < best_cost:
                    best = [r[:] for r in current]
                    best_cost = current_cost
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            it += 1
        T *= alpha
    return best, best_cost


# --------------------------------------- ALNS ---------------------------------------

SEED = 42  # graine pour garantir la reproductivité

rnd = random.Random(SEED)

# Helpers


def solution_cost(routes, D):
    total = 0.0
    for r in routes:
        if not r:
            continue
        total += D[0][r[0] - 1]  # départ dépôt
        for a, b in zip(r, r[1:]):
            total += D[a - 1][b - 1]  # arcs internes
        total += D[r[-1] - 1][0]  # retour dépôt
    return total


# Vérifie la compatibilité client / camion
def compatible(client, truck_index, node_type, truck_types, compat_dict=None):
    truck_id = truck_index + 1
    if compat_dict is not None:
        return bool(compat_dict.get((client, truck_id), 0))
    return node_type.get(client, None) == truck_types[truck_index]


# Vérifie la faisabilité d’insertion
def feasible_insert(route, pos, client, demand_dict, Q, truck_index):
    truck_id = truck_index + 1
    current_load = sum(demand_dict[c] for c in route)
    return (current_load + demand_dict[client]) <= Q[truck_id]


# Calcule le surcoût généré suite l'insertion d'un client à une pos donnée
# |-> Recalcule coût local de la route avant/après insertion
def insertion_delta(route, pos, client, D):
    if not route:
        return D[0][client - 1] + D[client - 1][0]
    old_cost = 0.0
    old_cost += D[0][route[0] - 1]
    for a, b in zip(route, route[1:]):
        old_cost += D[a - 1][b - 1]
    old_cost += D[route[-1] - 1][0]
    new = route[:pos] + [client] + route[pos:]
    new_cost = 0.0
    new_cost += D[0][new[0] - 1]
    for a, b in zip(new, new[1:]):
        new_cost += D[a - 1][b - 1]
    new_cost += D[new[-1] - 1][0]
    return new_cost - old_cost


# Destruct  & réparation


def flatten_routes(routes):
    return [c for r in routes for c in r]


# Destruction aléatoire -> détruit aléatoirement n clients dans notre ensemble de routes
def random_removal(routes, n_remove):
    allc = flatten_routes(routes)
    if not allc:
        return [r[:] for r in routes], []
    n_remove = min(n_remove, len(allc))
    removed = set(rnd.sample(allc, n_remove))
    new_routes = [[c for c in r if c not in removed] for r in routes]
    return new_routes, list(removed)


# Suppression des clients similaires
def shaw_removal(
    routes, n_remove, coords, demand, D, node_type, alpha=1.0, beta=0.1, gamma=0.5
):
    allc = flatten_routes(routes)
    if not allc:
        return [r[:] for r in routes], []
    seed = rnd.choice(allc)
    removed = {seed}
    max_remove = min(n_remove, len(allc))
    # sélection des plus similaires
    while len(removed) < max_remove:
        cand = [c for c in allc if c not in removed]
        if not cand:
            break
        rel = []
        for c in cand:
            # use D matrix: D indices 0-based
            d_xy = D[seed - 1][c - 1]
            d_dem = abs(demand[seed] - demand[c])
            d_type = 0 if node_type[seed] == node_type[c] else 1
            score = alpha * d_xy + beta * d_dem + gamma * d_type
            rel.append((score, c))
        rel.sort(key=lambda x: x[0])
        pick = int(rnd.random() ** 2 * len(rel))
        removed.add(rel[pick][1])
    new_routes = [[c for c in r if c not in removed] for r in routes]
    return new_routes, list(removed)


# insertion de chaque client au meilleur coût d’insertion possible
def greedy_insert(
    routes, removed, demand, D, Q, node_type, truck_types, compat_dict=None
):
    routes = [r[:] for r in routes]
    n_trucks = len(routes)
    for c in removed:
        best = None
        best_inc = float("inf")
        for k in range(n_trucks):
            # compat?
            if not compatible(c, k, node_type, truck_types, compat_dict):
                continue
            r = routes[k]
            for pos in range(len(r) + 1):
                if not feasible_insert(r, pos, c, demand, Q, k):
                    continue
                inc = insertion_delta(r, pos, c, D)
                if inc < best_inc:
                    best_inc = inc
                    best = (k, pos)
        if best is None:
            # impossible d'insérer ce client
            return None
        k, pos = best
        routes[k].insert(pos, c)
    return routes


# Choisit entre les 2 meilleures insertion possible
def regret2_insert(
    routes, removed, demand, D, Q, node_type, truck_types, compat_dict=None
):
    routes = [r[:] for r in routes]
    n_trucks = len(routes)
    R = set(removed)

    while R:
        best_client = None
        best_regret = -1.0
        best_place = None

        for c in list(R):
            inc_list = []
            places = []
            for k in range(n_trucks):
                if not compatible(c, k, node_type, truck_types, compat_dict):
                    continue
                r = routes[k]
                for pos in range(len(r) + 1):
                    if not feasible_insert(r, pos, c, demand, Q, k):
                        continue
                    inc = insertion_delta(r, pos, c, D)
                    inc_list.append(inc)
                    places.append((inc, k, pos))
            if not inc_list:
                continue
            inc_list.sort()
            best_inc = inc_list[0]
            second = inc_list[1] if len(inc_list) > 1 else inc_list[0]
            regret = second - best_inc
            if regret > best_regret:
                best_regret = regret
                best_client = c
                for inc, k, pos in places:
                    if abs(inc - best_inc) < 1e-12:
                        best_place = (k, pos)
                        break

        if best_client is None:
            return None

        R.remove(best_client)
        k, pos = best_place
        routes[k].insert(pos, best_client)

    return routes


# Alns Core


# ALNS principal
def alns(
    coords,
    demand,
    node_type,
    truck_types,
    D,
    Q,
    routes0,
    max_iter=2000,
    removal_rate=(0.15, 0.35),
    cooling=0.999,
    T0_factor=0.1,
    seed=SEED,
    verbose=True,
    compat_dict=None,
):
    rnd_local = random.Random(seed)
    n_clients = len(coords) - 1
    n_trucks = len(truck_types)
    nr_min = max(1, int(removal_rate[0] * n_clients))
    nr_max = max(nr_min, int(removal_rate[1] * n_clients))

    destroy_ops = [
        ("random", lambda r, n: random_removal(r, n)),
        ("shaw", lambda r, n: shaw_removal(r, n, coords, demand, D, node_type)),
    ]
    repair_ops = [
        (
            "greedy",
            lambda r, rem: greedy_insert(
                r, rem, demand, D, Q, node_type, truck_types, compat_dict
            ),
        ),
        (
            "regret2",
            lambda r, rem: regret2_insert(
                r, rem, demand, D, Q, node_type, truck_types, compat_dict
            ),
        ),
    ]

    wD = [1.0] * len(destroy_ops)
    wR = [1.0] * len(repair_ops)
    sD = [0.0] * len(destroy_ops)
    sR = [0.0] * len(repair_ops)
    uD = [0] * len(destroy_ops)
    uR = [0] * len(repair_ops)

    def pick(weights):
        tot = sum(weights)
        x = rnd_local.random() * tot
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if x <= acc:
                return i
        return len(weights) - 1

    def update_weights(reaction=0.2):
        for i in range(len(wD)):
            gain = (sD[i] / uD[i]) if uD[i] > 0 else 0.0
            wD[i] = (1 - reaction) * wD[i] + reaction * gain
        for i in range(len(wR)):
            gain = (sR[i] / uR[i]) if uR[i] > 0 else 0.0
            wR[i] = (1 - reaction) * wR[i] + reaction * gain
        for arr in (sD, sR, uD, uR):
            for j in range(len(arr)):
                arr[j] = 0 if isinstance(arr[j], int) else 0.0

    current = [r[:] for r in routes0]
    best = [r[:] for r in routes0]
    best_cost = solution_cost(best, D)
    current_cost = best_cost
    T = max(1e-9, T0_factor * max(1.0, best_cost))

    sigma1, sigma2, sigma3 = 5.0, 2.0, 0.5
    improvements = 0
    start = time.time()

    for it in range(1, max_iter + 1):
        nrem = rnd_local.randint(nr_min, nr_max)
        di = pick(wD)
        ri = pick(wR)
        cand_routes, removed = destroy_ops[di][1](current, nrem)
        cand_routes = repair_ops[ri][1](cand_routes, removed)
        if cand_routes is None:
            uD[di] += 1
            uR[ri] += 1
            sD[di] += 0.0
            sR[ri] += 0.0
            T *= cooling
            if it % 50 == 0:
                update_weights()
            continue

        cand_cost = solution_cost(cand_routes, D)

        if cand_cost < current_cost - 1e-9:
            current, current_cost = cand_routes, cand_cost
            sD[di] += sigma2
            sR[ri] += sigma2
            if cand_cost < best_cost - 1e-9:
                best, best_cost = [r[:] for r in cand_routes], cand_cost
                improvements += 1
                sD[di] += sigma1
                sR[ri] += sigma1
        else:
            delta = cand_cost - current_cost
            if rnd_local.random() < math.exp(-delta / T):
                current, current_cost = cand_routes, cand_cost
                sD[di] += sigma3
                sR[ri] += sigma3

        uD[di] += 1
        uR[ri] += 1
        T *= cooling

        if it % 50 == 0:
            update_weights()

    return (
        best,
        best_cost,
        {"improvements": improvements, "iterations": it},
    )


# Affichage


def print_solution_alns(routes, best_sol):
    total = 0.0
    for k, r in enumerate(routes, 1):
        print(f"  Truck {k} : {routes[k - 1]}")
    print(f"Cout total final = {best_cost}\n")


def parse_tsplib_vrp_with_types(instance_file_path):
    inst = vrplib.read_instance(instance_file_path)
    coords = inst["node_coord"]  # list of tuples (x,y)
    demands_list = inst["demand"]
    demand = {i: int(demands_list[i - 1]) for i in range(1, len(coords) + 1)}
    capacity_val = int(inst["capacity"])
    nb_nodes = int(inst["dimension"])
    depot_idx = (
        int(inst["depot"].item()) + 1
        if hasattr(inst["depot"], "item")
        else int(inst["depot"]) + 1
    )
    # truck types
    truck_types = (
        [
            int(inst["truck_type"][k - 1])
            for k in range(
                1,
                int(
                    re.search(
                        r"No of trucks[:\\s]*(\\d+)", inst.get("comment", "") or "1"
                    ).group(1)
                )
                + 1,
            )
        ]
        if False
        else None
    )
    try:
        truck_types = [
            int(inst["truck_type"][k - 1])
            for k in range(1, len(inst["truck_type"]) + 1)
        ]
    except Exception:
        truck_types = [1] * int(
            re.search(r"No of trucks[:\s]*(\d+)", inst.get("comment", "") or "1").group(
                1
            )
        )
    try:
        node_type = {
            i: int(inst["vehicle_type"][i - 1]) for i in range(1, nb_nodes + 1)
        }
    except Exception:
        node_type = {i: 1 for i in range(1, nb_nodes + 1)}

    # D matrix (euclidean on coords)
    def eucl(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    D = [[0.0] * nb_nodes for _ in range(nb_nodes)]
    for i in range(nb_nodes):
        for j in range(nb_nodes):
            D[i][j] = eucl(coords[i], coords[j])
    N_TRUCKS = len(truck_types)
    info = {"name": inst.get("name", ""), "comment": inst.get("comment", "")}
    return coords, demand, node_type, truck_types, D, capacity_val, N_TRUCKS, info


def initial_solution_with_types(
    coords, demand, node_type, truck_types, D, CAPACITY, rnd_local
):
    nb_nodes = len(coords)
    customers_list = [i for i in range(1, nb_nodes + 1) if i != 1]
    base_routes = initial_solution_greedy_for_type(
        customers_list, len(truck_types), capacity_val=CAPACITY
    )
    routes = [[] for _ in range(len(truck_types))]
    loads = [0] * len(truck_types)
    for r in base_routes:
        placed = False
        for k_idx in range(len(truck_types)):
            truck_id = k_idx + 1
            if loads[k_idx] + sum(demand[c] for c in r) <= CAPACITY:
                if all(node_type[c] == truck_types[k_idx] for c in r):
                    routes[k_idx] = r[:]
                    loads[k_idx] = sum(demand[c] for c in r)
                    placed = True
                    break
        if not placed:
            for c in r:
                for k_idx in range(len(truck_types)):
                    if (
                        loads[k_idx] + demand[c] <= CAPACITY
                        and node_type[c] == truck_types[k_idx]
                    ):
                        routes[k_idx].append(c)
                        loads[k_idx] += demand[c]
                        break
    return routes


# Initialise & fait tourner le ALNS
def initialisation_alns():
    node_type = {i: cust_type.get(i, 1) for i in range(1, len(coords) + 1)}
    node_type[1] = 0

    truck_types = [truck_type[k] for k in range(1, nb_trucks + 1)]

    D = dist

    CAPACITY = capacity
    N_TRUCKS = nb_trucks

    rnd_local = random.Random(SEED)
    routes0 = initial_solution_with_types(
        coords, d, node_type, truck_types, D, CAPACITY, rnd_local
    )

    return node_type, truck_types, D, routes0


# Génère les graphes représentant les routes empruntées lors des tournées
def show_graph(all_routes, total_cost_sum):
    depot_idx = depot - 1

    cmap = plt.get_cmap("tab20")
    plt.figure(figsize=(8, 6))

    for i, r in enumerate(all_routes):
        # convertir clients 1-based -> indices 0-based pour coords
        xs = (
            [coords[depot_idx][0]]
            + [coords[c - 1][0] for c in r]
            + [coords[depot_idx][0]]
        )
        ys = (
            [coords[depot_idx][1]]
            + [coords[c - 1][1] for c in r]
            + [coords[depot_idx][1]]
        )
        color = cmap(i % 20)
        plt.plot(xs, ys, marker="o", label=f"R{i + 1}", linewidth=1, color=color)
        plt.scatter(
            [coords[c - 1][0] for c in r],
            [coords[c - 1][1] for c in r],
            s=40,
            color=color,
        )

    # depot
    plt.scatter(
        [coords[depot_idx][0]],
        [coords[depot_idx][1]],
        c="k",
        marker="s",
        s=80,
        label="Depot",
    )
    plt.legend()
    plt.title(f"Total cost: {total_cost_sum:.1f}, routes: {len(all_routes)}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")  # conserve les proportions
    plt.tight_layout()
    plt.show()


def show_roads_2():
    truck_count_by_type = Counter(truck_type.values())

    all_routes_initial = []
    all_routes_final = []
    total_initial_cost = 0.0
    total_final_cost = 0.0

    for type_id, count in truck_count_by_type.items():
        (init_r, init_c, best_r, best_c) = solve_type_group(type_id, count)

        all_routes_initial.extend(init_r)
        all_routes_final.extend(best_r)

        total_initial_cost += init_c
        total_final_cost += best_c

    print("\nSolution initiale (gloutonne)")

    for i, r in enumerate(all_routes_initial, 1):
        print(f"Truck {i} : 1 {' '.join(map(str, r))} 1 | coût = {route_cost(r):.1f}")

    print(f"Coût total initial : {total_initial_cost:.1f}")

    print("\nSolution Simulated Annealing")

    for i, r in enumerate(all_routes_final, 1):
        print(f"Truck {i} : 1 {' '.join(map(str, r))} 1 | coût = {route_cost(r):.1f}")

    print(f"Coût total final : {total_final_cost:.1f}")

    return all_routes_final, total_final_cost


if __name__ == "__main__":
    # ---------------------------------------------------------
    # start = time.time()

    # prob, K, V, x = linear_program(nb_trucks, nodes, compat)
    # fonction_solver(prob)
    # total_cost_sum = pulp.value(prob.objective)
    # print(f"Objective val : {total_cost_sum}")
    # all_routes_PL = show_roads(K, V, x)
    # end = time.time()
    # print(f"Temps d’exécution : {end - start:.2f} s")
    # show_graph(all_routes_PL, total_cost_sum)

    # ---------------------------------------------------------
    # Le bloc ci-dessus concerne le PL -> commenter ce dernier lorsque instances > 50

    start = time.time()
    all_routes_SA, total_cost_sum = show_roads_2()
    end = time.time()
    print(f"Temps d’exécution : {end - start:.2f} s")
    show_graph(all_routes_SA, total_cost_sum)

    start = time.time()
    node_type, truck_types, D, routes0 = initialisation_alns()
    best_routes, best_cost, stats = alns(
        coords,
        d,
        node_type,
        truck_types,
        D,
        Q,
        routes0,
        max_iter=2000,
        removal_rate=(0.15, 0.35),
        cooling=0.999,
        T0_factor=0.1,
        seed=SEED,
        verbose=True,
        compat_dict=compat,
    )
    end = time.time()
    print_solution_alns(best_routes, best_cost)
    print(f"Temps d’exécution : {end - start:.2f} s")
    show_graph(best_routes, best_cost)

# coût optimal issu du .sol
cost_optimal = cost

# coût SA
_, cost_sa = show_roads_2()

# coût ALNS
cost_alns = best_cost

print(f"Coût optimal (fichier .sol) : {cost_optimal:.2f}")
if cost_sa is not None:
    print(f"Coût SA : {cost_sa:.2f}")
else:
    print("Coût SA : non calculé")

print(f"Coût ALNS : {cost_alns:.2f}")

print("\nGAP (%)")


def compute_gap(sol_cost, opt):
    return 100 * (sol_cost - opt) / opt


if cost_sa is not None:
    gap_sa = compute_gap(cost_sa, cost_optimal)
    print(f"GAP SA = {gap_sa:.3f} %")
else:
    print("GAP SA = N/A")

gap_alns = compute_gap(cost_alns, cost_optimal)
print(f"GAP ALNS = {gap_alns:.3f} %")
