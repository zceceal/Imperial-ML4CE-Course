# ------------------- GROUP INFO ----------------------
group_names     = ['Your Name']
cid_numbers     = ['00000000']
oral_assessment = [1]

# ------------------- IMPORTS -------------------------
import MLCE_CWBO2025.virtual_lab as virtual_lab
from MLCE_CWBO2025.gp_model import GP_model
import sobol_seq
import numpy as np
import random
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm


# -----------------------------------------------------
# HELPER 1: generate initial Sobol design (celltype fixed to 3)
# -----------------------------------------------------
def generate_initial_design_lab(n_init=6, seed=None):
    if seed is not None:
        np.random.seed(seed)

    sobol_points = sobol_seq.i4_sobol_generate(5, n_init)

    T_vals  = 30 + sobol_points[:, 0] * 10
    pH_vals = 6  + sobol_points[:, 1] * 2
    F1_vals = sobol_points[:, 2] * 50
    F2_vals = sobol_points[:, 3] * 50
    F3_vals = sobol_points[:, 4] * 50

    # FIX cell type to the best one found: 'celltype_3'
    cell_types = np.array(['celltype_3'] * n_init)

    X_init = []
    for i in range(n_init):
        X_init.append([
            float(T_vals[i]),
            float(pH_vals[i]),
            float(F1_vals[i]),
            float(F2_vals[i]),
            float(F3_vals[i]),
            str(cell_types[i])
        ])
    return X_init


# -----------------------------------------------------
# HELPER 2: convert LAB format -> GP numeric format (reduced: only 5 continuous inputs)
# -----------------------------------------------------
def X_lab_to_GP(X_lab):
    """
    Map lab-format X (with cell type) to GP numeric X:
    We use only the 5 continuous variables: [T, pH, F1, F2, F3].
    Cell type is fixed and not included as an input to the GP.
    """
    X_num = []
    for row in X_lab:
        cont = row[:5]  # T, pH, F1, F2, F3
        X_num.append(list(cont))
    return np.array(X_num, float)


# -----------------------------------------------------
# HELPER 3: wrapper for virtual lab
# -----------------------------------------------------
def objective_func(X_lab):
    return np.array(virtual_lab.conduct_experiment(X_lab)).reshape(-1,1)


# -----------------------------------------------------
# HELPER 4: acquisition function (Expected Improvement)
# -----------------------------------------------------
def acquisition_ei(X_cand_GP, gp, y_best, xi=0.1):
    """
    Expected Improvement for maximisation.

    X_cand_GP : (N, d) candidate points in GP numeric space
    gp        : trained GP_model instance
    y_best    : best observed objective so far (scalar)
    xi        : exploration parameter
    """
    N = X_cand_GP.shape[0]
    acq = np.zeros(N)

    for i in range(N):
        m, v = gp.GP_inference_np(X_cand_GP[i])
        mu = float(m[0])
        var = max(float(v[0]), 0.0)
        sigma = np.sqrt(var)

        if sigma < 1e-12:
            acq[i] = 0.0
            continue

        improvement = mu - y_best - xi
        z = improvement / sigma
        acq[i] = improvement * norm.cdf(z) + sigma * norm.pdf(z)

    return acq


# -----------------------------------------------------
# HELPER 5: Candidate generator for BO (celltype fixed to 3)
# -----------------------------------------------------
def generate_candidate_batch_lab(n_cand=300):
    sobol_points = sobol_seq.i4_sobol_generate(5, n_cand)

    T  = 30 + sobol_points[:,0]*10
    pH = 6  + sobol_points[:,1]*2
    F1 = sobol_points[:,2]*50
    F2 = sobol_points[:,3]*50
    F3 = sobol_points[:,4]*50

    # FIX cell type to 'celltype_3' for all candidates
    cell_types = np.array(['celltype_3'] * n_cand)

    X = []
    for i in range(n_cand):
        X.append([
            float(T[i]),
            float(pH[i]),
            float(F1[i]),
            float(F2[i]),
            float(F3[i]),
            cell_types[i]
        ])
    return X


# -----------------------------------------------------
# BO CLASS
# -----------------------------------------------------
class BO:
    def __init__(self,
                 max_iters=15,   # already enough to reach the max in your previous runs
                 batch_size=3,
                 n_init=6,
                 n_cand=500,
                 multi_hyper=1,
                 seed=None,
                 time_budget=None):

        # coursework requirement:
        start_time = datetime.timestamp(datetime.now())

        # RNG
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # logging
        self.X_lab = []
        self.Y     = []
        self.time  = []

        # ----- INITIAL DESIGN -----
        X_init = generate_initial_design_lab(n_init=n_init, seed=seed)
        Y_init = objective_func(X_init)

        self.X_lab = list(X_init)
        self.Y     = Y_init.flatten().tolist()

        # fit initial GP on reduced inputs
        X_GP = X_lab_to_GP(self.X_lab)
        self.gp = GP_model(X_GP, Y_init, 'RBF', multi_hyper, True)

        # record time for initial batch
        elapsed = datetime.timestamp(datetime.now()) - start_time
        self.time += [elapsed] + [0]*(n_init-1)
        start_time = datetime.timestamp(datetime.now())

        # ----- BO LOOP -----
        for it in range(max_iters):

            # optional time budget
            if (time_budget is not None) and (sum(self.time) > time_budget):
                print(f"Stopping because time_budget={time_budget}s reached.")
                break

            # propose new batch
            X_batch = self._propose_batch(n_cand, batch_size)

            # evaluate
            Y_batch = objective_func(X_batch).flatten().tolist()

            # store
            self.X_lab += X_batch
            self.Y     += Y_batch

            # refit GP on reduced inputs
            X_GP = X_lab_to_GP(self.X_lab)
            Y_np = np.array(self.Y).reshape(-1,1)
            self.gp = GP_model(X_GP, Y_np, 'RBF', multi_hyper, True)

            # timing
            elapsed = datetime.timestamp(datetime.now()) - start_time
            self.time += [elapsed] + [0]*(batch_size-1)
            start_time = datetime.timestamp(datetime.now())


    def _propose_batch(self, n_cand, batch_size):
        X_cand_lab = generate_candidate_batch_lab(n_cand)
        X_cand_GP  = X_lab_to_GP(X_cand_lab)

        # ε-greedy exploration: with some probability, pick random points
        explore_prob = 0.1  # 20% of the time, ignore EI and explore

        if np.random.rand() < explore_prob:
            # Pure exploration: select a random batch
            idx = np.random.choice(len(X_cand_lab), size=batch_size, replace=False)
        else:
            # Exploit/explore via EI
            y_best = max(self.Y)
            acq = acquisition_ei(X_cand_GP, self.gp, y_best, xi=0.1)
            idx = np.argsort(-acq)[:batch_size]

        return [X_cand_lab[i] for i in idx]


# -----------------------------------------------------
# BO EXECUTION BLOCK (REQUIRED)
# -----------------------------------------------------
BO_m = BO(
    max_iters=10,      # try 10 first; you can even test 6–8 later
    batch_size=5,
    n_init=6,
    n_cand=300,
    multi_hyper=1,
    seed=0,
    time_budget=120
)

# Inspect performance
Y_array = np.array(BO_m.Y)
best_so_far = np.maximum.accumulate(Y_array)
idx_first_best = np.argmax(best_so_far)

print("Best first reached at experiment index (0-based):", idx_first_best)
print("That’s experiment #", idx_first_best + 1, "in total.")
print("Total experiments:", len(BO_m.Y))
print("Max titre found:", max(BO_m.Y))
print("Best X:", BO_m.X_lab[np.argmax(BO_m.Y)])

# -----------------------------------------------------
# DENSE SOBOL SCAN TO CHECK GLOBALITY
# -----------------------------------------------------

# Number of random/Sobol points to scan
N_scan = 10000  # you can try 5000, 10000, 20000 depending on patience

# Generate Sobol points in 5D: [T, pH, F1, F2, F3]
sobol_points = sobol_seq.i4_sobol_generate(5, N_scan)

T_scan  = 30 + sobol_points[:, 0] * 10   # [30, 40]
pH_scan = 6  + sobol_points[:, 1] * 2    # [6, 8]
F1_scan = sobol_points[:, 2] * 50        # [0, 50]
F2_scan = sobol_points[:, 3] * 50        # [0, 50]
F3_scan = sobol_points[:, 4] * 50        # [0, 50]

# Fix cell type to 'celltype_3' as in your BO
cell_types_scan = np.array(['celltype_3'] * N_scan)

# Build lab-format design
X_scan = []
for i in range(N_scan):
    X_scan.append([
        float(T_scan[i]),
        float(pH_scan[i]),
        float(F1_scan[i]),
        float(F2_scan[i]),
        float(F3_scan[i]),
        cell_types_scan[i]
    ])

# Evaluate all points in the virtual lab (batch)
Y_scan = np.array(virtual_lab.conduct_experiment(X_scan)).reshape(-1)

# Get the best titre and corresponding X
idx_best_scan = np.argmax(Y_scan)
best_titre_scan = Y_scan[idx_best_scan]
best_X_scan = X_scan[idx_best_scan]

print("\n===== DENSE SOBOL SCAN RESULTS =====")
print("Number of scanned points:", N_scan)
print("Best titre from dense scan:", best_titre_scan)
print("Best X from dense scan:", best_X_scan)

# Compare to BO result
print("\n===== COMPARISON WITH BO OPTIMUM =====")
print("BO best titre:", max(BO_m.Y))
print("BO best X:", BO_m.X_lab[np.argmax(BO_m.Y)])
