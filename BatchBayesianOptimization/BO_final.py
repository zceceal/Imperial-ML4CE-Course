# MLCE_groupname_BO.py
# ------------------------------------------------------------
# MLCE Coursework 2025 – Batch Bayesian Optimisation
# ------------------------------------------------------------
# This script:
#   - Uses Gaussian Process helpers from gaussian_process2.py
#   - Runs batch BO with EI and a staged exploration→exploitation schedule
#   - Enumerates all 3 cell types per Sobol point
#   - Respects a 60 s time budget
#   - Prints the best titre and total optimisation time
#   - Uses at most 15 iterations with batch size 5
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
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import MLCE_CWBO2025.virtual_lab as virtual_lab

CONT_MIN = np.array([30.0, 6.0, 0.0, 0.0, 0.0])
CONT_MAX = np.array([40.0, 8.0, 50.0, 50.0, 50.0])
CONT_RANGE = CONT_MAX - CONT_MIN


# ============================================================
# Helper functions (inlined from gaussian_process2)
# ============================================================

def encode_cell_type(cell_str):
    if cell_str == 'celltype_1':
        return [1.0, 0.0, 0.0]
    if cell_str == 'celltype_2':
        return [0.0, 1.0, 0.0]
    if cell_str == 'celltype_3':
        return [0.0, 0.0, 1.0]
    raise ValueError(f"Unknown cell type: {cell_str}")


def X_lab_to_GP(X_lab):
    X_num = []
    for row in X_lab:
        cont = np.array(row[:5], dtype=float)
        cont_norm = (cont - CONT_MIN) / np.maximum(CONT_RANGE, 1e-12)
        cell_vec = encode_cell_type(row[5])
        X_num.append(list(cont_norm) + cell_vec)
    return np.array(X_num, dtype=float)


def objective_func(X_lab):
    y_list = virtual_lab.conduct_experiment(X_lab)
    return np.array(y_list).reshape(-1, 1)


def generate_initial_design_lab(n_init=6):
    sobol_points = sobol_seq.i4_sobol_generate(5, n_init)
    T_vals  = CONT_MIN[0] + sobol_points[:, 0] * CONT_RANGE[0]
    pH_vals = CONT_MIN[1] + sobol_points[:, 1] * CONT_RANGE[1]
    F_vals  = sobol_points[:, 2:5] * CONT_RANGE[2:5]

    base_cells = ['celltype_1', 'celltype_2', 'celltype_3']
    repeats = n_init // 3 + 1
    cell_types = (base_cells * repeats)[:n_init]

    X_init = []
    for i in range(n_init):
        X_init.append([
            float(T_vals[i]),
            float(pH_vals[i]),
            float(F_vals[i, 0]),
            float(F_vals[i, 1]),
            float(F_vals[i, 2]),
            cell_types[i]
        ])
    return X_init


# ============================================================
# GP_model inlined (from MLCE_CWBO2025.gp_model)
# ============================================================


class GP_model:

    def __init__(self, X, Y, kernel='RBF', multi_hyper=1, var_out=True):
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float)
        if self.Y.ndim == 1:
            self.Y = self.Y.reshape(-1, 1)

        self.kernel = kernel
        self.multi_hyper = max(1, multi_hyper)
        self.var_out = var_out

        self.n_point = self.X.shape[0]
        self.nx_dim = self.X.shape[1]
        self.ny_dim = self.Y.shape[1]

        self.X_mean = np.mean(self.X, axis=0)
        self.X_std = np.std(self.X, axis=0)
        self.Y_mean = np.mean(self.Y, axis=0)
        self.Y_std = np.std(self.Y, axis=0)

            # avoid division by zero
        self.X_std[self.X_std == 0.0] = 1.0
        self.Y_std[self.Y_std == 0.0] = 1.0

        self.X_norm = (self.X - self.X_mean) / self.X_std
        self.Y_norm = (self.Y - self.Y_mean) / self.Y_std

        self.hypopt, self.invKopt = self.determine_hyperparameters()

    def Cov_mat(self, kernel, X_norm, W, sf2):
        if kernel != 'RBF':
            raise ValueError(f"Unsupported kernel: {kernel}")
        dist = cdist(X_norm, X_norm, 'seuclidean', V=W)**2
        return sf2 * np.exp(-0.5 * dist)

    def calc_cov_sample(self, xnorm, Xnorm, ell, sf2):
        dist = cdist(Xnorm, xnorm.reshape(1, self.nx_dim), 'seuclidean', V=ell)**2
        return sf2 * np.exp(-0.5 * dist)

    def negative_loglikelihood(self, hyper, X, Y):
        W = np.exp(2 * hyper[:self.nx_dim])
        sf2 = np.exp(2 * hyper[self.nx_dim])
        sn2 = np.exp(2 * hyper[self.nx_dim + 1])

        K = self.Cov_mat(self.kernel, X, W, sf2)
        K = K + (sn2 + 1e-8) * np.eye(self.n_point)
        K = 0.5 * (K + K.T)
        L = np.linalg.cholesky(K)
        logdetK = 2 * np.sum(np.log(np.diag(L)))
        invLY = np.linalg.solve(L, Y)
        alpha = np.linalg.solve(L.T, invLY)
        NLL = np.dot(Y.T, alpha) + logdetK
        return NLL

    def determine_hyperparameters(self):
        X_norm, Y_norm = self.X_norm, self.Y_norm
        lb = np.array([-4.] * (self.nx_dim + 1) + [-8.])
        ub = np.array([4.] * (self.nx_dim + 1) + [-2.])
        bounds = np.stack([lb, ub], axis=1)
        multi_startvec = sobol_seq.i4_sobol_generate(self.nx_dim + 2, self.multi_hyper)

        hypopt = np.zeros((self.nx_dim + 2, self.ny_dim))
        invKopt = []

        for i in range(self.ny_dim):
            localsol = []
            localval = []
            for j in range(self.multi_hyper):
                hyp_init = lb + (ub - lb) * multi_startvec[j, :]
                res = minimize(
                    self.negative_loglikelihood,
                    hyp_init,
                    args=(X_norm, Y_norm[:, i]),
                    method='SLSQP',
                    bounds=bounds,
                    options={'disp': False, 'maxiter': 500},
                    tol=1e-12
                )
                localsol.append(res.x)
                localval.append(res.fun)

            minindex = int(np.argmin(localval))
            hypopt[:, i] = localsol[minindex]
            ellopt = np.exp(2.0 * hypopt[:self.nx_dim, i])
            sf2opt = np.exp(2.0 * hypopt[self.nx_dim, i])
            sn2opt = np.exp(2.0 * hypopt[self.nx_dim + 1, i]) + 1e-6
            Kopt = self.Cov_mat(self.kernel, X_norm, ellopt, sf2opt) \
                   + sn2opt * np.eye(self.n_point) \
                   + 1e-8 * np.eye(self.n_point)

            invKopt.append(np.linalg.solve(Kopt, np.eye(self.n_point)))

        return hypopt, invKopt

    def GP_inference_np(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        xnorm = (x - self.X_mean) / self.X_std
        mean = np.zeros(self.ny_dim)
        var = np.zeros(self.ny_dim)

        for i in range(self.ny_dim):
            invK = self.invKopt[i]
            hyper = self.hypopt[:, i]
            ellopt = np.exp(2 * hyper[:self.nx_dim])
            sf2opt = np.exp(2 * hyper[self.nx_dim])
            k = self.calc_cov_sample(xnorm, self.X_norm, ellopt, sf2opt)
            val_mean = np.matmul(np.matmul(k.T, invK), self.Y_norm[:, i])
            mean[i] = float(val_mean.squeeze())
            val_var = sf2opt - np.matmul(np.matmul(k.T, invK), k)
            var[i] = max(0.0, float(val_var.squeeze()))

        mean_sample = mean * self.Y_std + self.Y_mean
        var_sample = var * (self.Y_std ** 2)

        if self.var_out:
            return mean_sample, var_sample
        return mean_sample.flatten()[0]


# ============================================================
# 1) CANDIDATE GENERATOR FOR BO
# ============================================================
def generate_candidate_batch_lab(n_cand=400):
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
# 2) EXPECTED IMPROVEMENT (EI)
# ============================================================
def acquisition_ei(X_cand_GP, gp, y_best, xi=0.3):
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
# 3) BATCH BAYESIAN OPTIMISATION CLASS
# ============================================================
class BO:
    """
    Batch Bayesian Optimisation tuned for:
    - high max titre within 60 seconds,
    - staged exploration→exploitation over at most 15 iterations,
    - batch size fixed to 5 (parallel experiments),
    - better categorical exploration (all 3 cell types per Sobol point).

    Uses:
      - Initial Sobol design (from gaussian_process2)
      - RBF GP (GP_model) with multi_hyper restarts
      - EI acquisition with scheduled ξ
      - ε-greedy exploration with scheduled probability
      - light diversity within each batch
    """

    def __init__(self,
                 max_iters=15,       # coursework constraint: at most 15 iterations
                 batch_size=5,       # coursework constraint: batch size 5
                 n_init=6,
                 n_cand=5000,         # number of Sobol points -> 3 * n_cand candidates
                 multi_hyper=3,
                 seed=None,          # DEBUG ONLY – final submission: seed=None
                 time_budget=60):    # seconds

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

            # (a) Propose new batch (depends on iteration index it)
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

        For at most 15 iterations with batch size 5, we schedule:
          - Iterations 0-5: more exploratory EI (higher ξ, higher ε).
          - Iterations 8–14: more exploitative EI (smaller ξ), but still
            with some random exploration.
        Diversity in normalised GP space avoids selecting nearly-identical points.
        """
        # Generate candidates: 3 * n_cand points (all cell types)
        X_cand_lab = generate_candidate_batch_lab(n_cand=n_cand)
        X_cand_GP  = X_lab_to_GP(X_cand_lab)

        # --------- Exploration / exploitation schedule ----------
        if it < 5:
            # Early: explore but GP-guided
            explore_prob = 0.35      # stronger exploration early
            xi = 0.25
            min_dist = 0.20          # keep batches spread out

        elif it < 11:
            # Middle: balanced
            explore_prob = 0.30
            xi = 0.15
            min_dist = 0.15
        else:
            # Last 4 iterations: almost pure exploitation near current best
            explore_prob = 0.08
            xi = 0.02                # bump exploration parameter
            min_dist = 0.10          # still allow some spread
        
        # ε-greedy exploration
        if np.random.rand() < explore_prob:
            idx = np.random.choice(len(X_cand_lab), size=batch_size, replace=False)
            return [X_cand_lab[i] for i in idx]

        # EI-based selection with scheduled xi
        y_best = max(self.Y)
        acq = acquisition_ei(X_cand_GP, self.gp, y_best, xi=xi)
        order = np.argsort(-acq)  # best EI first

        # ---------- Diversity in normalised GP space ------------
        X_min = X_cand_GP.min(axis=0)
        X_max = X_cand_GP.max(axis=0)
        X_range = X_max - X_min + 1e-9
        X_norm = (X_cand_GP - X_min) / X_range

        chosen = []
        min_dist = 0.15  # slightly larger to spread batch points more

        for i in order:
            if len(chosen) == 0:
                chosen.append(i)
            else:
                dists = np.linalg.norm(X_norm[i] - X_norm[chosen], axis=1)
                if np.min(dists) > min_dist:
                    chosen.append(i)

            if len(chosen) == batch_size:
                break

        # If diversity rule did not fill batch, top-up with remaining best EI points
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
        max_iters=15,    # coursework constraint
        batch_size=5,    # coursework constraint
        n_init=6,
        n_cand=5000,      # 400 Sobol points -> 1200 candidates (3 cell types)
        multi_hyper=3,
        seed=None,          # DEBUG ONLY – remove / set None in submission
        time_budget=60
    )

    Y_array = np.array(BO_m.Y)
    best_idx = int(np.argmax(Y_array))

    print("\nBest titre found :", float(Y_array[best_idx]))
    print("Best X (LAB)     :", BO_m.X_lab[best_idx])
    print("Total experiments:", len(BO_m.X_lab))
    print("Approx. total optimisation time (s):", BO_m.total_time)
