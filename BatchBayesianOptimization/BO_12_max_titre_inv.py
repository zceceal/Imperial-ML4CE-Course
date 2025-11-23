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
from scipy.stats import norm   # for EI; in final submission you may switch to 'import scipy'

# -----------------------------------------------------
# HELPER 1: encode categorical variable (one-hot)
# -----------------------------------------------------
def encode_cell_type(cell_str):
    if cell_str == 'celltype_1':
        return [1.0, 0.0, 0.0]
    if cell_str == 'celltype_2':
        return [0.0, 1.0, 0.0]
    if cell_str == 'celltype_3':
        return [0.0, 0.0, 1.0]
    raise ValueError(f"Unknown cell type: {cell_str}")


# -----------------------------------------------------
# HELPER 2: convert LAB format -> GP numeric format (5 cont + 3 one-hot)
# -----------------------------------------------------
def X_lab_to_GP(X_lab):
    """
    Lab format:  [T, pH, F1, F2, F3, celltype_str]
    GP format:   [T, pH, F1, F2, F3, cell1, cell2, cell3]
    """
    X_num = []
    for row in X_lab:
        cont = row[:5]              # T, pH, F1, F2, F3
        oh   = encode_cell_type(row[5])
        X_num.append(list(cont) + oh)
    return np.array(X_num, float)


# -----------------------------------------------------
# HELPER 3: wrapper for virtual lab
# -----------------------------------------------------
def objective_func(X_lab):
    return np.array(virtual_lab.conduct_experiment(X_lab)).reshape(-1,1)


# -----------------------------------------------------
# HELPER 4: generate initial Sobol design (all 3 cell types)
# -----------------------------------------------------
def generate_initial_design_lab(n_init=6, seed=None):
    """
    Generate n_init Sobol points over the 5 continuous variables,
    and assign cell types such that all three appear at least once.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    sobol_points = sobol_seq.i4_sobol_generate(5, n_init)

    T_vals  = 30 + sobol_points[:, 0] * 10
    pH_vals = 6  + sobol_points[:, 1] * 2
    F1_vals = sobol_points[:, 2] * 50
    F2_vals = sobol_points[:, 3] * 50
    F3_vals = sobol_points[:, 4] * 50

    base_cells = ['celltype_1', 'celltype_2', 'celltype_3']
    cell_types = (base_cells * (n_init // 3 + 1))[:n_init]
    random.shuffle(cell_types)

    X_init = []
    for i in range(n_init):
        X_init.append([
            float(T_vals[i]),
            float(pH_vals[i]),
            float(F1_vals[i]),
            float(F2_vals[i]),
            float(F3_vals[i]),
            cell_types[i]
        ])
    return X_init


# -----------------------------------------------------
# HELPER 5: Candidate generator for BO (all 3 cell types)
# -----------------------------------------------------
def generate_candidate_batch_lab(n_cand=5000):
    """
    Generate n_cand Sobol points over the 5 continuous variables
    and assign random cell types from {1, 2, 3}.
    """
    sobol_points = sobol_seq.i4_sobol_generate(5, n_cand)

    T  = 30 + sobol_points[:,0]*10
    pH = 6  + sobol_points[:,1]*2
    F1 = sobol_points[:,2]*50
    F2 = sobol_points[:,3]*50
    F3 = sobol_points[:,4]*50

    cell_types = np.random.choice(['celltype_1','celltype_2','celltype_3'], size=n_cand)

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
# HELPER 6: acquisition function (Expected Improvement)
# -----------------------------------------------------
def acquisition_ei(X_cand_GP, gp, y_best, xi=0.05):
    """
    Expected Improvement for maximisation.

    X_cand_GP : (N, d) candidate points in GP numeric space
    gp        : trained GP_model instance
    y_best    : best observed objective so far (scalar)
    xi        : exploration parameter (bigger = more exploration)
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
# BO CLASS
# -----------------------------------------------------
class BO:
    def __init__(self,
                 max_iters=10,
                 batch_size=5,
                 n_init=6,
                 n_cand=5000,
                 multi_hyper=1,
                 seed=0,
                 time_budget=120):

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

        # fit initial GP on full 8D inputs
        X_GP = X_lab_to_GP(self.X_lab)
        self.gp = GP_model(X_GP, Y_init, 'RBF', multi_hyper, True)

        # record time for initial batch
        elapsed = datetime.timestamp(datetime.now()) - start_time
        self.time += [elapsed] + [0]*(n_init-1)
        start_time = datetime.timestamp(datetime.now())

        # ----- BO LOOP -----
        for it in range(max_iters):

            # stop if runtime exceeds time_budget (if given)
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

            # refit GP on full inputs
            X_GP = X_lab_to_GP(self.X_lab)
            Y_np = np.array(self.Y).reshape(-1,1)
            self.gp = GP_model(X_GP, Y_np, 'RBF', multi_hyper, True)

            # timing
            elapsed = datetime.timestamp(datetime.now()) - start_time
            self.time += [elapsed] + [0]*(batch_size-1)
            start_time = datetime.timestamp(datetime.now())


    def _propose_batch(self, n_cand, batch_size):
        # Generate candidates over full 6D space
        X_cand_lab = generate_candidate_batch_lab(n_cand)
        X_cand_GP  = X_lab_to_GP(X_cand_lab)

        # --------- NEW: compute anisotropic scaling from lengthscales ---------
        nx_dim = self.gp.nx_dim           # should be 8
        hypopt = self.gp.hypopt           # shape (nx_dim+2, ny_dim)
        ell = np.exp(2 * hypopt[:nx_dim, 0])  # lengthscales for output 0

        # avoid division by very small numbers
        ell_safe = np.maximum(ell, 1e-3)
        X_scaled = X_cand_GP / ell_safe   # anisotropic scaling

        # ----------------------------------------------------------------------
        # ε-greedy exploration: with some probability, pick random points
        explore_prob = 0.2  # 20% of the time, ignore EI and explore

        if np.random.rand() < explore_prob:
            idx = np.random.choice(len(X_cand_lab), size=batch_size, replace=False)
            return [X_cand_lab[i] for i in idx]

        # EI-based batch
        y_best = max(self.Y)
        acq = acquisition_ei(X_cand_GP, self.gp, y_best, xi=0.05)

        # Sort candidate indices by descending EI
        order = np.argsort(-acq)

        chosen = []
        min_dist = 1.0  # now interpreted in *scaled* space

        for i in order:
            if len(chosen) == 0:
                chosen.append(i)
            else:
                # use anisotropically scaled space for diversity
                dists = np.linalg.norm(X_scaled[i] - X_scaled[chosen], axis=1)
                if np.min(dists) > min_dist:
                    chosen.append(i)
            if len(chosen) == batch_size:
                break

        # If not enough points due to distance rule, fill remaining slots by EI
        if len(chosen) < batch_size:
            for i in order:
                if i not in chosen:
                    chosen.append(i)
                    if len(chosen) == batch_size:
                        break

        return [X_cand_lab[i] for i in chosen]


# -----------------------------------------------------
# BO EXECUTION BLOCK (REQUIRED)
# -----------------------------------------------------
BO_m = BO(
    max_iters=10,
    batch_size=5,
    n_init=6,
    n_cand=5000,
    multi_hyper=1,
    seed=0,
    time_budget=120
)

ell = np.exp(2 * BO_m.gp.hypopt[:BO_m.gp.nx_dim, 0])
labels = ["T", "pH", "F1", "F2", "F3", "cell1", "cell2", "cell3"]
print("\nLengthscales:")
for name, l in zip(labels, ell):
    print(f"{name}: {l:.3f}")

# Inspect performance (for debugging / analysis)
Y_array = np.array(BO_m.Y)
best_so_far = np.maximum.accumulate(Y_array)
idx_first_best = np.argmax(best_so_far)

print("Best first reached at experiment index (0-based):", idx_first_best)
print("That’s experiment #", idx_first_best + 1, "in total.")
print("Total experiments:", len(BO_m.Y))
print("Max titre found:", max(BO_m.Y))
print("Best X:", BO_m.X_lab[np.argmax(BO_m.Y)])