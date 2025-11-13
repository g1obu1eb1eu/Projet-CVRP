# ==============================================================
# ALNS pour CVRP AVEC TYPES DE VÉHICULES (TSPLIB .vrp)
# - Lit FILE_PATH (.vrp style H-n32-k6)
# - Prend en compte :
#     * coordonnées
#     * demandes
#     * CAPACITY
#     * VEHICLE_TYPE_SECTION (type de véhicule requis par client)
#     * TRUCK_TYPE_SECTION   (type de chaque camion)
# - ALNS avec compatibilité client–camion
# - Affiche solution initiale + solution ALNS
# ==============================================================

import math, random, time, re
from pathlib import Path

# >>>>>>>>>>>>>>>>>>>>>>>> A MODIFIER <<<<<<<<<<<<<<<<<<<<<<<<<<
# Path to CVRP instance directory
path_to_instances = Path(__file__).parent / "full_dataset"

# Path to specific instance & solution files

# Dataset 1
instance_file = path_to_instances / "H-n32-k6.vrp"
solution_file = path_to_instances / "H-n32-k6.sol"
SEED = 42
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ------------------------ Parser TSPLIB VRP -------------------
def parse_tsplib_vrp_with_types(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    
    def get_header(key, default=None):
        key = key.upper()
        for l in lines:
            if l.upper().startswith(key):
                return l.split(":",1)[1].strip()
        return default
    
    name     = get_header("NAME", "UNKNOWN")
    comment  = get_header("COMMENT", "")
    cap_str  = get_header("CAPACITY", "0")
    capacity = int(cap_str)
    
    # Nombre de camions : "-k6" dans le NAME ou "No of trucks: 6" dans le COMMENT
    m_k = re.search(r'-k(\d+)', name, re.IGNORECASE)
    m_c = re.search(r'No of trucks\s*:\s*(\d+)', comment, re.IGNORECASE)
    max_vehicles = int(m_k.group(1)) if m_k else (int(m_c.group(1)) if m_c else 999)
    
    def idx(tag):
        tag = tag.upper()
        for i,l in enumerate(lines):
            if l.upper() == tag:
                return i
        return -1
    
    i_coord  = idx("NODE_COORD_SECTION")
    i_demand = idx("DEMAND_SECTION")
    i_vtype  = idx("VEHICLE_TYPE_SECTION")
    i_ttype  = idx("TRUCK_TYPE_SECTION")
    i_depot  = idx("DEPOT_SECTION")
    
    # --- Coordonnées ---
    coords_dict = {}
    j = i_coord + 1
    while j < len(lines) and not lines[j].upper().endswith("SECTION") and lines[j].upper() != "EOF":
        parts = lines[j].split()
        if len(parts) >= 3 and parts[0].lstrip("-").isdigit():
            nid, x, y = int(parts[0]), float(parts[1]), float(parts[2])
            coords_dict[nid] = (x, y)
        j += 1
    
    # --- Demandes ---
    demand_dict = {}
    j = i_demand + 1
    while j < len(lines) and not lines[j].upper().endswith("SECTION") and lines[j].upper() != "EOF":
        parts = lines[j].split()
        if len(parts) >= 2 and parts[0].lstrip("-").isdigit():
            nid, d = int(parts[0]), int(parts[1])
            demand_dict[nid] = d
        j += 1
    
    # --- Types de clients (VEHICLE_TYPE_SECTION) ---
    node_type_dict = {}
    if i_vtype != -1:
        j = i_vtype + 1
        while j < len(lines) and not lines[j].upper().endswith("SECTION") and lines[j].upper() != "EOF":
            parts = lines[j].split()
            if len(parts) >= 2 and parts[0].lstrip("-").isdigit():
                nid, t = int(parts[0]), int(parts[1])
                node_type_dict[nid] = t
            j += 1
    
    # --- Types de camions (TRUCK_TYPE_SECTION) ---
    truck_type_dict = {}
    if i_ttype != -1:
        j = i_ttype + 1
        while j < len(lines) and not lines[j].upper().endswith("SECTION") and lines[j].upper() != "EOF":
            parts = lines[j].split()
            if len(parts) >= 2 and parts[0].lstrip("-").isdigit():
                tid, t = int(parts[0]), int(parts[1])
                truck_type_dict[tid] = t
            j += 1
    
    # --- Dépôt ---
    depot_ids = []
    if i_depot != -1:
        j = i_depot + 1
        while j < len(lines):
            if lines[j].upper() == "EOF" or lines[j] == "-1":
                break
            if lines[j].lstrip("-").isdigit():
                depot_ids.append(int(lines[j]))
            j += 1
    depot_id = depot_ids[0] if depot_ids else 1
    
    # Remap : mettre le dépôt en index 0
    ids = [depot_id] + [i for i in sorted(coords_dict.keys()) if i != depot_id]
    id_to_new = {old: new for new, old in enumerate(ids)}
    
    coords = [coords_dict[i] for i in ids]
    demand = [demand_dict.get(i, 0) for i in ids]
    node_type = [node_type_dict.get(i, 1) for i in ids]  # type 1 par défaut si absent
    node_type[0] = 0  # dépôt
    
    # Liste des types de camions triés par truck_id
    if truck_type_dict:
        truck_ids_sorted = sorted(truck_type_dict.keys())
        truck_types = [truck_type_dict[tid] for tid in truck_ids_sorted]
        n_trucks = len(truck_types)
    else:
        # fallback : flotte homogène
        truck_types = [1]*max_vehicles
        n_trucks = len(truck_types)
    
    # Matrice de distances euclidiennes
    n = len(coords)
    D = [[0.0]*n for _ in range(n)]
    for a in range(n):
        xa, ya = coords[a]
        for b in range(a+1, n):
            xb, yb = coords[b]
            d = math.hypot(xa - xb, ya - yb)
            D[a][b] = D[b][a] = d
    
    info = {
        "name": name,
        "comment": comment,
        "capacity": capacity,
        "n_nodes": n,
        "depot_old_id": depot_id,
        "n_trucks": n_trucks
    }
    return coords, demand, node_type, truck_types, D, capacity, n_trucks, info

# ------------------- Outils CVRP + types ----------------------
def route_load(route, demand):
    return sum(demand[i] for i in route)

def route_cost(route, D):
    if not route:
        return 0.0
    c = D[0][route[0]]
    for i in range(len(route)-1):
        c += D[route[i]][route[i+1]]
    c += D[route[-1]][0]
    return c

def solution_cost(routes, D):
    return sum(route_cost(r, D) for r in routes)

def insertion_delta(route, pos, client, D):
    if not route:
        return D[0][client] + D[client][0]
    if pos == 0:
        return D[0][client] + D[client][route[0]] - D[0][route[0]]
    if pos == len(route):
        return D[route[-1]][client] + D[client][0] - D[route[-1]][0]
    a, b = route[pos-1], route[pos]
    return D[a][client] + D[client][b] - D[a][b]

def feasible_insert(route, pos, client, demand, Q):
    return route_load(route, demand) + demand[client] <= Q

# compatibilité client c avec camion k
def compatible(client, truck_k, node_type, truck_types):
    t_client = node_type[client]
    t_truck  = truck_types[truck_k]
    # dépôt ou type 0 = on ignore
    if t_client == 0:
        return True
    return t_client == t_truck

# --------------------- Solution initiale ----------------------
def initial_solution_with_types(coords, demand, node_type, truck_types, D, Q, rnd):
    n_clients = len(coords) - 1
    n_trucks  = len(truck_types)
    # une route par camion
    routes = [[] for _ in range(n_trucks)]
    loads  = [0]*n_trucks
    
    cx, cy = coords[0]
    clients = list(range(1, n_clients+1))
    clients.sort(key=lambda i: math.atan2(coords[i][1]-cy, coords[i][0]-cx))
    
    for c in clients:
        best = None
        best_inc = float('inf')
        # on essaie tous les camions compatibles
        for k in range(n_trucks):
            if not compatible(c, k, node_type, truck_types):
                continue
            r = routes[k]
            # insertion en fin (simple) + vérification capacité
            pos = len(r)
            if feasible_insert(r, pos, c, demand, Q):
                inc = insertion_delta(r, pos, c, D)
                if inc < best_inc:
                    best_inc = inc
                    best = (k, pos)
        # si aucune insertion faisable niveau capacité, on force quand même la meilleure insertion
        if best is None:
            for k in range(n_trucks):
                if not compatible(c, k, node_type, truck_types):
                    continue
                r = routes[k]
                pos = len(r)
                inc = insertion_delta(r, pos, c, D)
                if inc < best_inc:
                    best_inc = inc
                    best = (k, pos)
        if best is None:
            # aucun camion compatible ? (ne devrait pas arriver si dataset cohérent)
            k = rnd.randrange(n_trucks)
            routes[k].append(c)
        else:
            k, pos = best
            routes[k].insert(pos, c)
    return routes

# ----------------- Destruction / Réparation -------------------
rnd = random.Random(SEED)

def flatten_routes(routes):
    return [c for r in routes for c in r]

def random_removal(routes, n_remove):
    allc = flatten_routes(routes)
    if not allc:
        return [r[:] for r in routes], []
    n_remove = min(n_remove, len(allc))
    removed = set(rnd.sample(allc, n_remove))
    new_routes = [[c for c in r if c not in removed] for r in routes]
    return new_routes, list(removed)

def shaw_removal(routes, n_remove, coords, demand, D, node_type, alpha=1.0, beta=0.1, gamma=0.5):
    allc = flatten_routes(routes)
    if not allc:
        return [r[:] for r in routes], []
    seed = rnd.choice(allc)
    removed = {seed}
    while len(removed) < min(n_remove, len(allc)):
        cand = [c for c in allc if c not in removed]
        if not cand:
            break
        rel = []
        for c in cand:
            d_xy = D[seed][c]
            d_dem = abs(demand[seed] - demand[c])
            d_type = 0 if node_type[seed] == node_type[c] else 1
            score = alpha*d_xy + beta*d_dem + gamma*d_type
            rel.append((score, c))
        rel.sort(key=lambda x: x[0])
        pick = int(rnd.random()**2 * len(rel))
        removed.add(rel[pick][1])
    new_routes = [[c for c in r if c not in removed] for r in routes]
    return new_routes, list(removed)

def greedy_insert(routes, removed, demand, D, Q, node_type, truck_types):
    routes = [r[:] for r in routes]
    n_trucks = len(routes)

    for c in removed:
        best = None
        best_inc = float('inf')

        for k in range(n_trucks):
            if not compatible(c, k, node_type, truck_types):
                continue
            r = routes[k]
            for pos in range(len(r) + 1):
                if feasible_insert(r, pos, c, demand, Q):
                    inc = insertion_delta(r, pos, c, D)
                    if inc < best_inc:
                        best_inc = inc
                        best = (k, pos)

        # AUCUNE position faisable => on signale l’échec
        if best is None:
            return None

        k, pos = best
        routes[k].insert(pos, c)

    return routes


def regret2_insert(routes, removed, demand, D, Q, node_type, truck_types):
    routes = [r[:] for r in routes]
    n_trucks = len(routes)
    R = set(removed)

    while R:
        best_client = None
        best_regret = -1.0
        best_place = None

        for c in R:
            inc_list = []
            places = []

            for k in range(n_trucks):
                if not compatible(c, k, node_type, truck_types):
                    continue
                r = routes[k]
                for pos in range(len(r) + 1):
                    if feasible_insert(r, pos, c, demand, Q):
                        inc = insertion_delta(r, pos, c, D)
                        inc_list.append(inc)
                        places.append((inc, k, pos))

            if not inc_list:
                # Pas de position faisable pour ce client
                continue

            inc_list.sort()
            best_inc = inc_list[0]
            second = inc_list[1] if len(inc_list) > 1 else inc_list[0]
            regret = second - best_inc

            if regret > best_regret:
                best_regret = regret
                best_client = c
                # retrouver la place correspondant au best_inc
                for inc, k, pos in places:
                    if abs(inc - best_inc) < 1e-12:
                        best_place = (k, pos)
                        break

        # Si on n’a trouvé aucun client insérable à ce tour → échec global
        if best_client is None:
            return None

        R.remove(best_client)
        k, pos = best_place
        routes[k].insert(pos, best_client)

    return routes


# ------------------------ ALNS core ---------------------------
def alns(coords, demand, node_type, truck_types, D, Q, routes0,
         max_iter=2000, removal_rate=(0.15,0.35), cooling=0.999, T0_factor=0.1, seed=42, verbose=True):
    rnd = random.Random(seed)
    n_clients = len(coords) - 1
    n_trucks  = len(truck_types)
    nr_min = max(1, int(removal_rate[0]*n_clients))
    nr_max = max(nr_min, int(removal_rate[1]*n_clients))
    
    destroy_ops = [
        ("random", lambda r,n: random_removal(r, n)),
        ("shaw",   lambda r,n: shaw_removal(r, n, coords, demand, D, node_type))
    ]
    repair_ops = [
        ("greedy",  lambda r,rem: greedy_insert(r, rem, demand, D, Q, node_type, truck_types)),
        ("regret2", lambda r,rem: regret2_insert(r, rem, demand, D, Q, node_type, truck_types))
    ]
    
    wD = [1.0]*len(destroy_ops); wR = [1.0]*len(repair_ops)
    sD = [0.0]*len(destroy_ops); sR = [0.0]*len(repair_ops)
    uD = [0]*len(destroy_ops);   uR = [0]*len(repair_ops)
    
    def pick(weights):
        tot = sum(weights); x = rnd.random()*tot; acc = 0.0
        for i,w in enumerate(weights):
            acc += w
            if x <= acc: return i
        return len(weights)-1
    
    def update_weights(reaction=0.2):
        for i in range(len(wD)):
            gain = (sD[i]/uD[i]) if uD[i] > 0 else 0.0
            wD[i] = (1-reaction)*wD[i] + reaction*gain
        for i in range(len(wR)):
            gain = (sR[i]/uR[i]) if uR[i] > 0 else 0.0
            wR[i] = (1-reaction)*wR[i] + reaction*gain
        for arr in (sD, sR, uD, uR):
            for j in range(len(arr)):
                arr[j] = 0 if isinstance(arr[j], int) else 0.0
    
    current = [r[:] for r in routes0]
    best    = [r[:] for r in routes0]
    best_cost = solution_cost(best, D)
    current_cost = best_cost
    T = max(1e-9, T0_factor * max(1.0, best_cost))
    
    sigma1, sigma2, sigma3 = 5.0, 2.0, 0.5
    improvements = 0
    start = time.time()
    
    for it in range(1, max_iter+1):
        nrem = rnd.randint(nr_min, nr_max)
        di = pick(wD); ri = pick(wR)
        cand_routes, removed = destroy_ops[di][1](current, nrem)
        cand_routes = repair_ops[ri][1](cand_routes, removed)
        cand_cost = solution_cost(cand_routes, D)
        
        if cand_cost < current_cost - 1e-9:
            current, current_cost = cand_routes, cand_cost
            sD[di] += sigma2; sR[ri] += sigma2
            if cand_cost < best_cost - 1e-9:
                best, best_cost = [r[:] for r in cand_routes], cand_cost
                improvements += 1
                sD[di] += sigma1; sR[ri] += sigma1
        else:
            delta = cand_cost - current_cost
            if rnd.random() < math.exp(-delta / T):
                current, current_cost = cand_routes, cand_cost
                sD[di] += sigma3; sR[ri] += sigma3
        
        uD[di] += 1; uR[ri] += 1
        T *= cooling
        
        if it % 50 == 0:
            update_weights()
        
        if verbose and it % 200 == 0:
            print(f"[it {it}] best={best_cost:.2f} current={current_cost:.2f} T={T:.4f}")
    
    elapsed = time.time() - start
    return best, best_cost, {"time_sec": elapsed, "improvements": improvements, "iterations": it}

# ------------------------ Affichage ---------------------------
def print_solution(routes, demand, D, title="Solution"):
    print(title)
    total = 0.0
    for k, r in enumerate(routes, 1):
        ld = route_load(r, demand)
        ct = route_cost(r, D)
        total += ct
        print(f"  Truck {k:02d} | load={ld:>3} | cost={ct:7.2f} | {r}")
    print(f"TOTAL COST = {total:.2f}\n")

# ============================ RUN =============================

coords, demand, node_type, truck_types, D, CAPACITY, N_TRUCKS, info = parse_tsplib_vrp_with_types(instance_file)

print(f"[Dataset] {info['name']} | {info['comment']}")
print(f"CAPACITY={CAPACITY}  | N_clients={len(coords)-1} | N_trucks={N_TRUCKS}")
print(f"Types de camions: {truck_types}")
print(f"Types des premiers clients: {[node_type[i] for i in range(1, min(6, len(node_type)))]}\n")

rnd = random.Random(SEED)
routes0 = initial_solution_with_types(coords, demand, node_type, truck_types, D, CAPACITY, rnd)
print_solution(routes0, demand, D, title="=== SOLUTION INITIALE (types respectés) ===")

best_routes, best_cost, stats = alns(
    coords, demand, node_type, truck_types, D, CAPACITY, routes0,
    max_iter=2000,
    removal_rate=(0.15,0.35),
    cooling=0.999,
    T0_factor=0.1,
    seed=SEED,
    verbose=True
)

print_solution(best_routes, demand, D, title="=== SOLUTION ALNS (types respectés) ===")
print(f"Best cost: {best_cost:.2f} | Time: {stats['time_sec']:.2f}s | Improvements: {stats['improvements']} | Iters: {stats['iterations']}")
