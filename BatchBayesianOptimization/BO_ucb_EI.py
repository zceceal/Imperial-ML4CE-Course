# MLCE_groupname_BO.py
# ------------------------------------------------------------
# MLCE Coursework 2025 – Batch Bayesian Optimisation
# ------------------------------------------------------------
# This script:
#   - Uses Gaussian Process helpers from gaussian_process2.py
#   - Runs batch BO with EI (exploitative settings)
#   - Enumerates all 3 cell types per Sobol point
#   - Respects a 60 s time budget
#   - Prints the best titre and total optimisation time
# ------------------------------------------------------------

# ============ GROUP INFO (EDIT THESE) =======================
group_names     = ['Your Name']
cid_numbers     = ['00000000']
oral_assessment = [1]


# ============ IMPORTS =======================================
import numpy as np
import random
from datetime import datetime
from scipy.stats import norm
import sobol_seq

import MLCE_CWBO2025.virtual_lab as virtual_lab
from MLCE_CWBO2025.gp_model import GP_model

# -----------------------------------------------------------
# 1) ENCODING: cell type (categorical) -> one-hot vector
# -----------------------------------------------------------

def encode_cell_type(cell_str):
    """
    Convert a cell type string into a numeric one-hot vector.
    - 'celltype_1' -> [1.0, 0.0, 0.0]
    - 'celltype_2' -> [0.0, 1.0, 0.0]
    - 'celltype_3' -> [0.0, 0.0, 1.0]
    """
    if cell_str == 'celltype_1':
        return [1.0, 0.0, 0.0]
    elif cell_str == 'celltype_2':
        return [0.0, 1.0, 0.0]
    elif cell_str == 'celltype_3':
        return [0.0, 0.0, 1.0]
    else:
        raise ValueError(f"Unknown cell type: {cell_str}")


# -----------------------------------------------------------
# 2) CONVERSION: LAB format -> GP numeric format
# -----------------------------------------------------------

def X_lab_to_GP(X_lab):
    """
    Convert LAB-format X (used by the virtual lab) into numeric GP-format X.

    LAB format (each row):
        [T, pH, F1, F2, F3, celltype_str]

    GP format (each row):
        [T, pH, F1, F2, F3, cell1, cell2, cell3]
    where [cell1, cell2, cell3] is a one-hot vector.

    Input:
        X_lab : list of rows in LAB format
    Output:
        NumPy array of shape (N, 8)
    """
    X_num = []  # empty list to collect numeric rows

    for row in X_lab:
        cont = row[:5]             # T, pH, F1, F2, F3
        cell_str = row[5]          # celltype string
        cell_oh = encode_cell_type(cell_str)
        X_num.append(list(cont) + cell_oh)

    return np.array(X_num, dtype=float)


# -----------------------------------------------------------
# 3) OBJECTIVE: wrapper around the virtual lab
# -----------------------------------------------------------

def objective_func(X_lab):
    """
    Evaluate the virtual bioprocess on a batch of LAB-format points.

    Input:
        X_lab : list of LAB-format rows
                [[T, pH, F1, F2, F3, celltype_str], ...]
    Output:
        Y : NumPy array of shape (N, 1) with the titre values
    """
    y_list = virtual_lab.conduct_experiment(X_lab)  # list of length N with scalar titres
    return np.array(y_list).reshape(-1, 1)


# -----------------------------------------------------------
# 4) INITIAL DESIGN: Sobol points + balanced cell types
# -----------------------------------------------------------

def generate_initial_design_lab(n_init=6):
    """
    Generate n_init initial experiments in LAB format.

    - Use a Sobol sequence for the 5 continuous variables.
    - Assign cell types so that all three appear at least once.

    Returns:
        X_initial_lab : list of LAB-format points
    """
    # (a) Sobol samples in [0, 1]^5
    sobol_points = sobol_seq.i4_sobol_generate(5, n_init)  # shape: (n_init, 5)

    # (b) Scale to experimental ranges
    T_vals  = 30.0 + sobol_points[:, 0] * 10.0   # T in [30, 40]
    pH_vals = 6.0  + sobol_points[:, 1] * 2.0    # pH in [6, 8]
    F1_vals = 0.0  + sobol_points[:, 2] * 50.0   # F1 in [0, 50]
    F2_vals = 0.0  + sobol_points[:, 3] * 50.0   # F2 in [0, 50]
    F3_vals = 0.0  + sobol_points[:, 4] * 50.0   # F3 in [0, 50]

    # (c) Balanced cell types
    base_cells = ['celltype_1', 'celltype_2', 'celltype_3']
    repeats = n_init // 3 + 1
    cell_types = (base_cells * repeats)[:n_init]
    cell_types = np.array(cell_types)
    # In final coursework, avoid fixed seeding; shuffling is optional.
    # np.random.shuffle(cell_types)

    # (d) Build LAB-format list
    X_initial_lab = []
    for i in range(n_init):
        X_initial_lab.append([
            float(T_vals[i]),
            float(pH_vals[i]),
            float(F1_vals[i]),
            float(F2_vals[i]),
            float(F3_vals[i]),
            str(cell_types[i])
        ])

    return X_initial_lab


# -----------------------------------------------------------
# 5) OPTIONAL: helper to build and test a GP
# -----------------------------------------------------------

def build_trained_gp(n_init=6, multi_hyper=1
                     ):
    """
    Create an initial design, run the virtual lab, and fit a GP_model.

    Returns:
        Xtrain_lab : list of LAB-format points         (length n_init)
        Ytrain     : (n_init, 1) NumPy array of titres
        Xtrain_GP  : (n_init, 8) NumPy array in GP numeric space
        gp         : trained GP_model instance
    """
    # 1) initial design
    Xtrain_lab = generate_initial_design_lab(n_init=n_init)
    # 2) evaluate virtual lab
    Ytrain = objective_func(Xtrain_lab)
    # 3) convert to GP space
    Xtrain_GP = X_lab_to_GP(Xtrain_lab)
    # 4) fit GP with RBF kernel and multi-start hyperparameters
    gp = GP_model(
        Xtrain_GP,
        Ytrain,
        kernel='RBF',
        multi_hyper=multi_hyper,
        var_out=True
    )
    return Xtrain_lab, Ytrain, Xtrain_GP, gp


# ============================================================
# 1) CANDIDATE GENERATOR FOR BO
# ============================================================
def generate_candidate_batch_lab(n_cand=5000, x_best_lab=None, it=None):
    """
    Generate candidate points in LAB format.

    Strategy:
    - Draw n_cand Sobol points for the 5 continuous variables (T, pH, F1, F2, F3).
    - For EACH Sobol point, enumerate ALL 3 cell types:
        => total number of candidates = 3 * n_cand

    This is more systematic than assigning random cell types and helps
    the BO explore the categorical space more efficiently.
    """
    # Sobol samples in [0,1]^5
    sobol_points = sobol_seq.i4_sobol_generate(5, n_cand)  # (n_cand, 5)

    X_cand_lab = []

    for i in range(n_cand):
        # Scale to experimental ranges
        T  = 30.0 + sobol_points[i, 0] * 10.0   # T in [30, 40]
        pH = 6.0  + sobol_points[i, 1] * 2.0    # pH in [6, 8]
        F1 =        sobol_points[i, 2] * 50.0   # F1 in [0, 50]
        F2 =        sobol_points[i, 3] * 50.0   # F2 in [0, 50]
        F3 =        sobol_points[i, 4] * 50.0   # F3 in [0, 50]

        # Enumerate ALL cell types for this continuous point
        for cell in ['celltype_1', 'celltype_2', 'celltype_3']:
            X_cand_lab.append([
                float(T),
                float(pH),
                float(F1),
                float(F2),
                float(F3),
                cell
            ])

    # Total candidates: 3 * n_cand
    return X_cand_lab


# ============================================================
# 2a) EXPECTED IMPROVEMENT (EI) – MORE EXPLOITATIVE
# ============================================================
def acquisition_ei(X_cand_GP, gp, y_best, xi=0.05):
    """
    Expected Improvement for maximisation.

    Parameters:
        X_cand_GP : (N, d) candidate points (GP numeric space)
        gp        : trained GP_model instance
        y_best    : best observed titre so far
        xi        : exploration parameter (small = more exploitation)

    With a small xi, EI focuses more on high predicted mean while still
    rewarding uncertainty through sigma.
    """
    N = X_cand_GP.shape[0]
    acq = np.zeros(N)

    for i in range(N):
        x_i = X_cand_GP[i]
        mean_vec, var_vec = gp.GP_inference_np(x_i)

        mu   = float(mean_vec[0])
        var  = max(float(var_vec[0]), 0.0)
        sigma = np.sqrt(var)

        if sigma < 1e-12:
            acq[i] = 0.0
            continue

        improvement = mu - y_best - xi
        z = improvement / sigma
        acq[i] = improvement * norm.cdf(z) + sigma * norm.pdf(z)

    return acq

# ============================================================
# 2b) UCB ACQUISITION (for exploration-focused phases)
# ============================================================
def acquisition_ucb(X_cand_GP, gp, beta=2.0):
    """
    Upper Confidence Bound (UCB) acquisition for maximisation.

    a(x) = mu(x) + sqrt(beta) * sigma(x)
    where beta controls exploration strength.
    """
    N = X_cand_GP.shape[0]
    acq = np.zeros(N)

    for i in range(N):
        x_i = X_cand_GP[i]
        mean_vec, var_vec = gp.GP_inference_np(x_i)

        mu   = float(mean_vec[0])
        var  = max(float(var_vec[0]), 0.0)
        sigma = np.sqrt(var)

        # If nearly no uncertainty, UCB ~ mean
        acq[i] = mu + np.sqrt(beta) * sigma

    return acq


# ============================================================
# 3) BATCH BAYESIAN OPTIMISATION CLASS
# ============================================================
class BO:
    """
    Batch Bayesian Optimisation tuned for:
    - high max titre within 60 seconds,
    - relatively exploitative behaviour (small xi),
    - better categorical exploration (all 3 cell types per Sobol point).

    Uses:
      - Initial Sobol design (from gaussian_process2)
      - RBF GP (GP_model) with multi_hyper restarts
      - EI acquisition (xi small)
      - ε-greedy exploration (small epsilon)
      - light diversity within each batch
    """

    def __init__(self,
                 max_iters=15,       # upper bound; time_budget may stop earlier
                 batch_size=5,
                 n_init=6,
                 n_cand=5000,        # number of Sobol points -> 3 * n_cand candidates
                 multi_hyper=3,
                 seed=None,         # DEBUG ONLY – final submission: seed=None
                 time_budget=60):   # seconds

        # Record start time of the whole optimisation
        self.start_wall = datetime.timestamp(datetime.now())

        # For debugging only; do NOT set seed in final submission
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Logs
        self.X_lab = []   # LAB-format experiments
        self.Y     = []   # observed titres
        self.time  = []   # elapsed time markers

        # ----------------- INITIAL DESIGN --------------------
        start_batch = datetime.timestamp(datetime.now())

        X_init = generate_initial_design_lab(n_init=n_init)
        Y_init = objective_func(X_init).flatten().tolist()

        self.X_lab = list(X_init)
        self.Y     = list(Y_init)

        X_GP_init = X_lab_to_GP(self.X_lab)
        Y_np_init = np.array(self.Y).reshape(-1, 1)

        # GP: multi_hyper = 1 for speed / more aggressive fit
        self.gp = GP_model(
            X_GP_init,
            Y_np_init,
            kernel='RBF',
            multi_hyper=multi_hyper,
            var_out=True
        )

        elapsed = datetime.timestamp(datetime.now()) - start_batch
        self.time += [elapsed] + [0.0] * (n_init - 1)

        # ------------------- BO LOOP -------------------------
        for it in range(max_iters):

            # Stop if we exceed the global time budget
            if (time_budget is not None) and (sum(self.time) > time_budget):
                print(f"Stopping because time_budget = {time_budget}s was reached.")
                break

            start_batch = datetime.timestamp(datetime.now())

            # (a) Propose new batch
            X_batch_lab = self._propose_batch(
                n_cand=n_cand,
                batch_size=batch_size,
                it=it
            )

            # (b) Evaluate batch
            Y_batch = objective_func(X_batch_lab).flatten().tolist()

            # (c) Add data
            self.X_lab += X_batch_lab
            self.Y     += Y_batch

            # (d) Refit GP on all data
            X_GP_all = X_lab_to_GP(self.X_lab)
            Y_np_all = np.array(self.Y).reshape(-1, 1)

            self.gp = GP_model(
                X_GP_all,
                Y_np_all,
                kernel='RBF',
                multi_hyper=multi_hyper,
                var_out=True
            )

            # (e) Update time log
            elapsed = datetime.timestamp(datetime.now()) - start_batch
            self.time += [elapsed] + [0.0] * (batch_size - 1)
            print(f"After batch {it + 1}, best titre so far: {max(self.Y):.3f}")

        # store total optimisation time
        self.total_time = sum(self.time)


    # --------------------------------------------------------
    # INTERNAL: propose one batch
    # --------------------------------------------------------
    def _propose_batch(self, n_cand, batch_size, it):
        """
        Propose a batch of new LAB-format points.

        Strategy:
        - Early iterations (it < 2): UCB (exploration-focused).
        - Later iterations (it >= 2): EI (exploitative).
        - In both phases:
            * ε-exploration: random among top-acquisition candidates.
            * Diversity: min-distance constraint in candidate space.
        """
        # Generate candidates: 3 * n_cand points (all cell types)
        X_cand_lab = generate_candidate_batch_lab(n_cand=n_cand)
        X_cand_GP  = X_lab_to_GP(X_cand_lab)

        # ------------------ phase-dependent settings ------------------
        if it < 2:
            # EARLY PHASE: UCB + more exploration
            mode         = "ucb"
            beta         = 5.0      # stronger exploration in UCB
            explore_prob = 0.20     # more randomisation early
            xi           = None     # not used for UCB
            min_dist     = 0.30
        else:
            # LATER PHASE: EI + more exploitation
            mode         = "ei"
            beta         = None     # not used for EI
            explore_prob = 0.02
            xi           = 0.01
            min_dist     = 0.15
        # --------------------------------------------------------------

        # --- Compute acquisition values for all candidates ---
        if mode == "ucb":
            acq = acquisition_ucb(X_cand_GP, self.gp, beta=beta)
        else:
            y_best = max(self.Y)
            acq = acquisition_ei(X_cand_GP, self.gp, y_best, xi=xi)

        # ---------------- ε-exploration but acquisition-guided ----------
        if np.random.rand() < explore_prob:
            # pick randomly among high-acquisition candidates (e.g. top 20%)
            frac_top = 0.2
            n_top = max(batch_size, int(frac_top * len(acq)))
            top_idx = np.argsort(-acq)[:n_top]   # indices of top acq points

            chosen_idx = np.random.choice(top_idx, size=batch_size, replace=False)
            return [X_cand_lab[i] for i in chosen_idx]
        # ----------------------------------------------------------------

        # -------- GREEDY selection + diversity based on acquisition -----
        order = np.argsort(-acq)  # best acquisition first

        # Light diversity using distances in normalised space
        X_min = X_cand_GP.min(axis=0)
        X_max = X_cand_GP.max(axis=0)
        X_range = X_max - X_min + 1e-9
        X_norm = (X_cand_GP - X_min) / X_range

        chosen = []

        for i in order:
            if len(chosen) == 0:
                chosen.append(i)
            else:
                dists = np.linalg.norm(X_norm[i] - X_norm[chosen], axis=1)
                if np.min(dists) > min_dist:
                    chosen.append(i)

            if len(chosen) == batch_size:
                break

        # Top-up if needed
        if len(chosen) < batch_size:
            for i in order:
                if i not in chosen:
                    chosen.append(i)
                    if len(chosen) == batch_size:
                        break

        return [X_cand_lab[i] for i in chosen]

# ============================================================
# 4) EXECUTION BLOCK (LOCAL TEST ONLY)
# ============================================================
if __name__ == "__main__":
    # For debugging: use a seed; for final submission: seed=None
    BO_m = BO(
        max_iters=15,
        batch_size=5,
        n_init=6,
        n_cand=5000,   # 400 is a good compromise
        multi_hyper=3,  # see GP section below
        seed=None,
        time_budget=None  # let iteration budget be the constraint
    )


    Y_array = np.array(BO_m.Y)
    best_idx = int(np.argmax(Y_array))

    print("\nBest titre found :", float(Y_array[best_idx]))
    print("Best X (LAB)     :", BO_m.X_lab[best_idx])
    print("Total experiments:", len(BO_m.X_lab))
    print("Approx. total optimisation time (s):", BO_m.total_time) 
