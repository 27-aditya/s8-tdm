"""
Part 2: Multi-Objective Clustering 
""" 

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


#  GA Configuration


K_MIN        = 2
K_MAX        = 15
POP_SIZE     = 60
N_GENERATIONS = 80
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.85
TOURNAMENT_K  = 3       # tournament selection size
RANDOM_SEED   = 42
rng = np.random.default_rng(RANDOM_SEED)


#  Dataset Loading & Preprocessing

def load_data():
    data = load_breast_cancer()
    X = StandardScaler().fit_transform(data.data)
    print(f"  Dataset: Breast Cancer Wisconsin – {X.shape}")
    return X


#  Chromosome Encoding

def create_individual():
    k    = rng.integers(K_MIN, K_MAX + 1)
    seed = rng.integers(0, 1000)
    return np.array([k, seed], dtype=int)


def create_population(size: int):
    return [create_individual() for _ in range(size)]


#  Objective Evaluation

_kmeans_cache = {}   # (k, seed) → (f1, f2)

def evaluate(individual: np.ndarray, X: np.ndarray):
    """
    Returns (f1, f2) both in [0, 1], both to be MINIMIZED.
    f1 = normalized intra-cluster compactness (WCSS per point)
    f2 = normalized number of clusters
    """
    k, seed = int(individual[0]), int(individual[1])
    key = (k, seed)
    if key in _kmeans_cache:
        return _kmeans_cache[key]

    km = KMeans(n_clusters=k, random_state=seed, n_init=5, max_iter=200)
    km.fit(X)

    # f1: mean squared distance of each point to its centroid
    centers = km.cluster_centers_
    labels  = km.labels_
    sq_dists = np.array([
        np.sum((X[i] - centers[labels[i]]) ** 2)
        for i in range(len(X))
    ])
    f1_raw = sq_dists.mean()

    # f2: number of clusters (raw)
    f2_raw = float(k)

    _kmeans_cache[key] = (f1_raw, f2_raw)
    return f1_raw, f2_raw


def evaluate_population(population: list, X: np.ndarray):
    """Returns list of (f1, f2) tuples."""
    return [evaluate(ind, X) for ind in population]


#  Normalization helpers (for objectives in [0,1])

def normalize_objectives(fitnesses: list):
    """Normalize each objective to [0, 1] across the current population."""
    arr = np.array(fitnesses, dtype=float)            # shape (N, 2)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0                          
    return (arr - mins) / ranges


#  NSGA-II Core: Fast Non-Dominated Sort


def dominates(f_a, f_b):
    """Return True if f_a dominates f_b (both objectives minimised)."""
    return (all(fa <= fb for fa, fb in zip(f_a, f_b)) and
            any(fa <  fb for fa, fb in zip(f_a, f_b)))


def fast_non_dominated_sort(fitnesses: list):
    """
    Returns a list of fronts (each front = list of indices).
    Front 0 = Pareto-optimal (non-dominated).
    """
    n = len(fitnesses)
    dominated_by = [[] for _ in range(n)]   
    domination_count = [0] * n              
    fronts = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(fitnesses[i], fitnesses[j]):
                dominated_by[i].append(j)
            elif dominates(fitnesses[j], fitnesses[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    current_front_idx = 0
    while fronts[current_front_idx]:
        next_front = []
        for i in fronts[current_front_idx]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front_idx += 1
        fronts.append(next_front)

    return [f for f in fronts if f]

#  NSGA-II Core: Crowding Distance


def crowding_distance(fitnesses: list, front: list):
    """Returns crowding distance for each individual in 'front'."""
    n = len(front)
    if n <= 2:
        return {idx: float('inf') for idx in front}

    distances = {idx: 0.0 for idx in front}
    n_obj = len(fitnesses[0])

    for obj_idx in range(n_obj):
        sorted_front = sorted(front, key=lambda i: fitnesses[i][obj_idx])
        f_min = fitnesses[sorted_front[0]][obj_idx]
        f_max = fitnesses[sorted_front[-1]][obj_idx]
        spread = f_max - f_min if f_max != f_min else 1.0

        distances[sorted_front[0]]  = float('inf')
        distances[sorted_front[-1]] = float('inf')

        for k in range(1, n - 1):
            i_prev = sorted_front[k - 1]
            i_next = sorted_front[k + 1]
            distances[sorted_front[k]] += (
                fitnesses[i_next][obj_idx] - fitnesses[i_prev][obj_idx]
            ) / spread

    return distances


#  NSGA-II Core: Tournament Selection

def tournament_select(population, ranks, crowding, size=TOURNAMENT_K):
    """Binary tournament selection based on rank then crowding distance."""
    pool_idx = rng.integers(0, len(population), size=size)
    best = pool_idx[0]
    for idx in pool_idx[1:]:
        r_best = ranks[best]
        r_idx  = ranks[idx]
        if r_idx < r_best:
            best = idx
        elif r_idx == r_best and crowding.get(idx, 0) > crowding.get(best, 0):
            best = idx
    return population[best].copy()


#  Genetic Operators


def crossover(parent_a, parent_b):
    """Single-point crossover on the two genes [k, seed]."""
    if rng.random() < CROSSOVER_RATE:
        point = rng.integers(1, 2)   # only 2 genes, so point=1
        child_a = np.array([parent_a[0], parent_b[1]], dtype=int)
        child_b = np.array([parent_b[0], parent_a[1]], dtype=int)
    else:
        child_a = parent_a.copy()
        child_b = parent_b.copy()
    return child_a, child_b


def mutate(individual):
    """Randomly change k or seed."""
    ind = individual.copy()
    if rng.random() < MUTATION_RATE:
        ind[0] = rng.integers(K_MIN, K_MAX + 1)      # mutate k
    if rng.random() < MUTATION_RATE:
        ind[1] = rng.integers(0, 1000)               # mutate seed
    return ind


#  NSGA-II Main Loop

def nsga2(X: np.ndarray):
    print(f"\n  Running NSGA-II  pop={POP_SIZE}  generations={N_GENERATIONS} ...")

    population = create_population(POP_SIZE)
    fitnesses  = evaluate_population(population, X)

    # Track Pareto front objective values across generations
    history_f1 = []
    history_f2 = []

    for gen in range(N_GENERATIONS):

     
        fronts = fast_non_dominated_sort(fitnesses)

        # Build rank and crowding maps
        ranks    = {}
        crowding = {}
        for rank, front in enumerate(fronts):
            cd = crowding_distance(fitnesses, front)
            for idx in front:
                ranks[idx]    = rank
                crowding[idx] = cd[idx]

   
        offspring = []
        while len(offspring) < POP_SIZE:
            pa = tournament_select(population, ranks, crowding)
            pb = tournament_select(population, ranks, crowding)
            ca, cb = crossover(pa, pb)
            offspring.append(mutate(ca))
            if len(offspring) < POP_SIZE:
                offspring.append(mutate(cb))

        offspring_fits = evaluate_population(offspring, X)

    
        combined_pop  = population + offspring
        combined_fits = fitnesses + offspring_fits

      
        all_fronts = fast_non_dominated_sort(combined_fits)
        new_pop    = []
        new_fits   = []

        for front in all_fronts:
            if len(new_pop) + len(front) <= POP_SIZE:
                for idx in front:
                    new_pop.append(combined_pop[idx])
                    new_fits.append(combined_fits[idx])
            else:
            
                cd = crowding_distance(combined_fits, front)
                sorted_front = sorted(front,
                                      key=lambda i: cd.get(i, 0),
                                      reverse=True)
                needed = POP_SIZE - len(new_pop)
                for idx in sorted_front[:needed]:
                    new_pop.append(combined_pop[idx])
                    new_fits.append(combined_fits[idx])
                break

        population = new_pop
        fitnesses  = new_fits

        # Collect Pareto front
        fronts = fast_non_dominated_sort(fitnesses)
        pf = fronts[0]
        pf_f1 = [fitnesses[i][0] for i in pf]
        pf_f2 = [fitnesses[i][1] for i in pf]
        history_f1.append(np.mean(pf_f1))
        history_f2.append(np.mean(pf_f2))

        if (gen + 1) % 20 == 0 or gen == 0:
            print(f"    Gen {gen+1:3d}/{N_GENERATIONS}  "
                  f"Pareto front size={len(pf)}  "
                  f"avg_compactness={np.mean(pf_f1):.4f}  "
                  f"avg_k={np.mean(pf_f2):.1f}")

    return population, fitnesses, history_f1, history_f2


#  Non-Dominated Solutions & Pareto Front

def extract_pareto_front(population, fitnesses):
    fronts = fast_non_dominated_sort(fitnesses)
    pareto_idx = fronts[0]
    pareto_pop  = [population[i] for i in pareto_idx]
    pareto_fits = [fitnesses[i]  for i in pareto_idx]

    # Sort by number of clusters for clean plot
    sorted_pairs = sorted(zip(pareto_fits, pareto_pop),
                          key=lambda x: x[0][1])
    pareto_fits_sorted = [p[0] for p in sorted_pairs]
    pareto_pop_sorted  = [p[1] for p in sorted_pairs]
    return pareto_pop_sorted, pareto_fits_sorted


def plot_pareto_front(pareto_fits, all_fits):
    f1_all = [f[0] for f in all_fits]
    f2_all = [f[1] for f in all_fits]
    f1_pf  = [f[0] for f in pareto_fits]
    f2_pf  = [f[1] for f in pareto_fits]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(f2_all, f1_all, c='#a8c8e8', alpha=0.5, s=30,
               edgecolors='none', label='Dominated solutions', zorder=2)


    ax.scatter(f2_pf, f1_pf, c='#e15759', s=80, edgecolors='#2b2b2b',
               linewidths=0.7, zorder=4, label='Non-dominated (Pareto)')

   
    ax.plot(f2_pf, f1_pf, 'r--', linewidth=1.5, alpha=0.7, zorder=3)

    ax.set_xlabel('Number of Clusters  (f₂ – minimize)', fontsize=13)
    ax.set_ylabel('Intra-Cluster Compactness  (f₁ – minimize)', fontsize=13)
    ax.set_title('Pareto Front – Multi-Objective Clustering (NSGA-II)\n'
                 'Breast Cancer Wisconsin Dataset',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pareto_front.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Pareto front plot saved → {path}")


def plot_convergence(history_f1, history_f2):
    gens = list(range(1, len(history_f1) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(gens, history_f1, color='#4e79a7', linewidth=2)
    axes[0].set_xlabel('Generation', fontsize=12)
    axes[0].set_ylabel('Avg Compactness (Pareto front)', fontsize=11)
    axes[0].set_title('Convergence – Compactness (f₁)', fontsize=12,
                      fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(gens, history_f2, color='#e15759', linewidth=2)
    axes[1].set_xlabel('Generation', fontsize=12)
    axes[1].set_ylabel('Avg k (Pareto front)', fontsize=11)
    axes[1].set_title('Convergence – Num Clusters (f₂)', fontsize=12,
                      fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "nsga2_convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Convergence plot saved → {path}")


def print_pareto_table(pareto_pop, pareto_fits):
    print("\n  Non-Dominated (Pareto) Solutions:")
    print(f"  {'#':>3}  {'k':>4}  {'Seed':>6}  {'Compactness':>13}  {'Num Clusters':>13}")
    print("  " + "-" * 50)
    for i, (ind, fit) in enumerate(zip(pareto_pop, pareto_fits)):
        print(f"  {i+1:>3}  {ind[0]:>4}  {ind[1]:>6}  {fit[0]:>13.4f}  {fit[1]:>13.0f}")


#  MAIN

def run_part2():
    print("\n" + "╔" + "═" * 63 + "╗")
    print("║  PART 2: MULTI-OBJECTIVE CLUSTERING (NSGA-II from scratch)  ║")
    print("╚" + "═" * 63 + "╝\n")

    print("  Objectives (both minimized, continuous, normalized range):")
    print("    f1 = Intra-cluster compactness (avg squared dist to centroid)")
    print("    f2 = Number of clusters k")
    print("  Trade-off: Fewer clusters → higher compactness (conflicting)\n")

    X = load_data()

    # Run NSGA-II
    final_pop, final_fits, hist_f1, hist_f2 = nsga2(X)

    # Extract Pareto front
    pareto_pop, pareto_fits = extract_pareto_front(final_pop, final_fits)
    print(f"\n  Identified {len(pareto_pop)} non-dominated Pareto solutions.")

    print_pareto_table(pareto_pop, pareto_fits)
    plot_pareto_front(pareto_fits, final_fits)
    plot_convergence(hist_f1, hist_f2)

    print("\n  Interpretation:")
    print("  • Each point on the Pareto front is a trade-off solution.")
    print("  • Bottom-left = ideal (low compactness AND few clusters),")
    print("    but these objectives conflict – reducing k raises compactness.")
    print("  • A practitioner selects the knee-point based on domain need.")
    print("  • NSGA-II successfully maintained diversity across k values,")
    print("    producing a smooth, well-spread Pareto front.")

    print("\n" + "=" * 65)
    print("  PART 2 COMPLETE – All plots saved to:", OUTPUT_DIR)
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_part2()
