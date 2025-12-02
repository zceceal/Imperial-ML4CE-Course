# ------------------- GROUP INFO ----------------------
group_names     = ['Your Name']
cid_numbers     = ['00000000']
oral_assessment = [1]

# ------------------- IMPORTS -------------------------
import MLCE_CWBO2025.virtual_lab as virtual_lab
import numpy as np
import scipy
import random
import sobol_seq
import time
from datetime import datetime

CONT_MIN = np.array([30.0, 6.0, 0.0, 0.0, 0.0])
CONT_MAX = np.array([40.0, 8.0, 50.0, 50.0, 50.0])
CONT_RANGE = CONT_MAX - CONT_MIN

# -----------------------------------------------------
# HELPER: Simple GP model (RBF kernel)
# -----------------------------------------------------
class GP_model:
    """
    Lightweight Gaussian Process with RBF kernel.
    Meant to replace MLCE_CWBO2025.gp_model.GP_model using only allowed packages.
    """

    def __init__(self, X, Y, kernel_type='RBF', multi_hyper=1, optimise=True):
        # X: (N, d), Y: (N, 1)
        self.X = np.asarray(X, dtype=float)
        self.Y = np.asarray(Y, dtype=float).reshape(-1, 1)
        self.n, self.d = self.X.shape

        self.kernel_type = kernel_type
        self.multi_hyper = max(1, int(multi_hyper))

        # Hyperparameters in log-space
        # lengthscales (one per dimension), signal variance, noise std
        self.log_lengthscales = np.zeros(self.d)        # log(1.0)
        self.log_signal       = np.log(1.0)
        self.log_noise        = np.log(1e-3)            # small initial noise

        # Will be set in _build_cache()
        self.L      = None
        self.alpha  = None

        if optimise:
            self._optimise_hypers()

        self._build_cache()

    # ---------------- GP internals --------------------
    def _get_hyper_vector(self):
        return np.concatenate([
            self.log_lengthscales,
            np.array([self.log_signal, self.log_noise])
        ])

    def _set_hyper_vector(self, theta):
        theta = np.asarray(theta, float)
        self.log_lengthscales = theta[:self.d]
        self.log_signal       = theta[self.d]
        self.log_noise        = theta[self.d + 1]

    def _rbf_kernel(self, X1, X2, lengthscales, signal_var):
        # X1: (n1, d), X2: (n2, d)
        # lengthscales: (d,), signal_var: scalar
        X1_scaled = X1 / lengthscales
        X2_scaled = X2 / lengthscales
        sqdist = (
            np.sum(X1_scaled**2, axis=1).reshape(-1, 1)
            + np.sum(X2_scaled**2, axis=1).reshape(1, -1)
            - 2.0 * np.dot(X1_scaled, X2_scaled.T)
        )
        return signal_var * np.exp(-0.5 * sqdist)

    def _kernel_train(self):
        lengthscales = np.exp(self.log_lengthscales)
        signal_var   = np.exp(self.log_signal) ** 2
        noise_std    = np.exp(self.log_noise)
        K = self._rbf_kernel(self.X, self.X, lengthscales, signal_var)
        K += (noise_std**2 + 1e-8) * np.eye(self.n)
        return K

    def _build_cache(self):
        # Build K, its Cholesky, and alpha = K^{-1} y
        K = self._kernel_train()
        self.L, lower = scipy.linalg.cho_factor(K, lower=True, overwrite_a=True, check_finite=False)
        self.alpha = scipy.linalg.cho_solve((self.L, lower), self.Y, check_finite=False)

    def _neg_log_marginal_likelihood(self, theta):
        # Objective for hyperparameter optimisation
        self._set_hyper_vector(theta)
        try:
            K = self._kernel_train()
            L, lower = scipy.linalg.cho_factor(K, lower=True, overwrite_a=False, check_finite=False)
        except np.linalg.LinAlgError:
            # If numerical issues, penalise this theta
            return 1e25

        alpha = scipy.linalg.cho_solve((L, lower), self.Y, check_finite=False)
        # 0.5 * y^T K^{-1} y + sum(log(diag(L))) + 0.5 * n * log(2*pi)

        quad = self.Y.T.dot(alpha)          # shape (1, 1)
        quad_scalar = quad.ravel()[0]       # extract scalar safely

        data_fit = 0.5 * quad_scalar
        complexity = np.sum(np.log(np.diag(L)))
        const = 0.5 * self.n * np.log(2.0 * np.pi)
        return data_fit + complexity + const


    def _optimise_hypers(self):
        # Simple multi-start optimisation of log marginal likelihood
        best_theta = self._get_hyper_vector()
        best_nll   = self._neg_log_marginal_likelihood(best_theta)

        bounds = []
        # lengthscales: log(0.05) to log(5.0)
        for _ in range(self.d):
            bounds.append((np.log(0.05), np.log(5.0)))
        # signal: log(0.1) to log(10)
        bounds.append((np.log(0.1), np.log(10.0)))
        # noise std: log(1e-5) to log(0.5)
        bounds.append((np.log(1e-5), np.log(0.5)))

        for _ in range(self.multi_hyper):
            theta0 = self._get_hyper_vector() + np.random.normal(scale=0.5, size=self.d + 2)
            res = scipy.optimize.minimize(
                self._neg_log_marginal_likelihood,
                theta0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50}
            )
            if res.success and res.fun < best_nll:
                best_nll = res.fun
                best_theta = res.x

        self._set_hyper_vector(best_theta)

    # ---------------- Prediction interface --------------------
    def predict(self, X_star):
        """
        X_star: (N*, d)
        Returns:
            mean: (N*, 1)
            var:  (N*, 1)
        """
        X_star = np.asarray(X_star, dtype=float)
        if X_star.ndim == 1:
            X_star = X_star.reshape(1, -1)

        lengthscales = np.exp(self.log_lengthscales)
        signal_var   = np.exp(self.log_signal) ** 2

        K_s = self._rbf_kernel(X_star, self.X, lengthscales, signal_var)  # (N*, n)

        # mean = K_s K^{-1} y = K_s alpha
        mean = K_s.dot(self.alpha)  # (N*, 1)

        # var = k(x*,x*) - v^T v
        v = scipy.linalg.cho_solve((self.L, True), K_s.T, check_finite=False)  # (n, N*)
        # prior variance at each test point is signal_var
        var = signal_var - np.sum(K_s.T * v, axis=0)  # shape (N*,)

        var = np.maximum(var, 1e-12)  # numerical stability
        return mean.reshape(-1, 1), var.reshape(-1, 1)

    def GP_inference_np(self, x_star):
        """
        Single-point prediction to mimic original interface.
        x_star: (d,) or (1,d)
        Returns:
            mean: np.array shape (1,)
            var:  np.array shape (1,)
        """
        m, v = self.predict(np.asarray(x_star, dtype=float).reshape(1, -1))
        return m.flatten(), v.flatten()


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
    X_num = []
    for row in X_lab:
        T, pH, F1, F2, F3, cell = row
        # scale to ~[0, 1]
        T_scaled  = (T  - 30) / 10
        pH_scaled = (pH - 6)  / 2
        F1_scaled = F1 / 50
        F2_scaled = F2 / 50
        F3_scaled = F3 / 50
        oh = encode_cell_type(cell)
        X_num.append([T_scaled, pH_scaled, F1_scaled, F2_scaled, F3_scaled] + oh)
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
    n_init must be <= 6 to satisfy coursework.
    """
    # randomness comes from global RNG state; no per-call seeding to avoid fixed seeds

    # Sobol over the continuous variables: n_init distinct points
    sobol_points = sobol_seq.i4_sobol_generate(5, n_init)

    T_vals  = 30 + sobol_points[:, 0] * 10
    pH_vals = 6  + sobol_points[:, 1] * 2
    F1_vals = sobol_points[:, 2] * 50
    F2_vals = sobol_points[:, 3] * 50
    F3_vals = sobol_points[:, 4] * 50

    # ensure all three cell types appear at least once
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
def generate_candidate_batch_lab(n_base=5000, cell_types=None, x_best_lab=None, it=None):
    """
    Generate candidates by taking n_base Sobol points in the continuous space
    and pairing each with all 3 cell types -> 3 * n_base total candidates.
    """
    if cell_types is None:
        cell_types = ['celltype_1', 'celltype_2', 'celltype_3']

    sobol_points = sobol_seq.i4_sobol_generate(5, n_base)

    T  = CONT_MIN[0] + sobol_points[:,0]*CONT_RANGE[0]
    pH = CONT_MIN[1] + sobol_points[:,1]*CONT_RANGE[1]
    F1 = CONT_MIN[2] + sobol_points[:,2]*CONT_RANGE[2]
    F2 = CONT_MIN[3] + sobol_points[:,3]*CONT_RANGE[3]
    F3 = CONT_MIN[4] + sobol_points[:,4]*CONT_RANGE[4]

    X = []
    for i in range(n_base):
        cont_part = [float(T[i]), float(pH[i]), float(F1[i]), float(F2[i]), float(F3[i])]
        for cell in cell_types:
            X.append(cont_part + [cell])

    if (x_best_lab is not None) and (it is not None) and (it >= 2):
        base = np.array(x_best_lab[:5], dtype=float)
        base_norm = (base - CONT_MIN) / np.maximum(CONT_RANGE, 1e-9)
        radius = max(0.05, 0.35 * (0.9 ** max(0, it - 1)))
        for _ in range(max(5, n_base // 5)):
            noise = np.random.normal(0.0, radius, size=5)
            cont_norm = np.clip(base_norm + noise, 0.0, 1.0)
            cont = CONT_MIN + cont_norm * CONT_RANGE
            X.append([
                float(cont[0]),
                float(cont[1]),
                float(cont[2]),
                float(cont[3]),
                float(cont[4]),
                x_best_lab[5]
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
    # Vectorised prediction
    mu, var = gp.predict(X_cand_GP)
    mu   = mu.flatten()
    var  = var.flatten()
    sigma = np.sqrt(var)

    # Avoid division by zero
    sigma_safe = np.where(sigma < 1e-12, 1e-12, sigma)

    improvement = mu - y_best - xi
    z = improvement / sigma_safe

    norm = scipy.stats.norm
    ei = improvement * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # For points with almost zero variance, EI ~ 0
    ei[sigma < 1e-12] = 0.0
    return ei


# -----------------------------------------------------
# BO CLASS
# -----------------------------------------------------
class BO:
    def __init__(self,
                 max_iters=10,
                 batch_size=5,
                 n_init=6,
                 n_base=5000,
                 multi_hyper=3,
                 seed=None,
                 time_budget=60):

        # coursework requirement: first line in __init__
        start_time = datetime.timestamp(datetime.now())

        # Safety checks: coursework constraints
        max_iters   = min(max_iters, 15)   # max 15
        batch_size  = min(batch_size, 5)   # max 5
        n_init      = min(n_init, 6)       # max 6

        # RNG
        # Leave RNG unseeded to avoid deterministic trajectories

        # logging
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.X_lab = []
        self.Y     = []
        self.time  = []
        self.best_progress = []

        # ----- INITIAL DESIGN -----
        X_init = generate_initial_design_lab(n_init=n_init, seed=None)
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
        self.best_progress.append(max(self.Y))

        # ----- BO LOOP -----
        for it in range(max_iters):

            # stop if runtime exceeds time_budget (if given)
            if (time_budget is not None) and (sum(self.time) > time_budget):
                print(f"Stopping because time_budget={time_budget}s reached.")
                break

            # propose new batch
            X_batch = self._propose_batch(n_base, batch_size, it)

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
            current_best = max(self.Y)
            self.best_progress.append(current_best)
            print(f"After batch {it + 1}, best titre so far: {current_best:.3f}")
            print(f"After batch {it + 1}, best titre so far: {max(self.Y):.3f}")


    def _propose_batch(self, n_base, batch_size, it):
        allowed_cells = self._allowed_cells(it)
        x_best_lab, y_best = self._current_best()

        X_cand_lab = generate_candidate_batch_lab(
            n_base,
            cell_types=allowed_cells,
            x_best_lab=x_best_lab,
            it=it
        )
        X_cand_GP  = X_lab_to_GP(X_cand_lab)

        # Exploration probability decays over iterations
        max_iters = self.max_iters
        stagnating = self._is_stagnating(window=2, tol=0.2)
        explore_prob = max(0.05, 0.35 * (1 - it / max_iters))
        if stagnating:
            explore_prob = min(0.6, explore_prob + 0.2)

        if np.random.rand() < explore_prob:
            idx = np.random.choice(len(X_cand_lab), size=batch_size, replace=False)
            return [X_cand_lab[i] for i in idx]

        # EI exploration parameter also decays
        xi_start, xi_end = 0.18, 0.02
        if max_iters > 1:
            xi = xi_start + (xi_end - xi_start) * (it / (max_iters-1))
        else:
            xi = xi_end
        if stagnating:
            xi *= 1.5

        acq = acquisition_ei(X_cand_GP, self.gp, y_best, xi=xi)

        # Sort candidate indices by descending EI
        order = np.argsort(-acq)

        chosen = []
        min_dist = 0.3 if not stagnating else 0.15

        for i in order:
            if len(chosen) == 0:
                chosen.append(i)
            else:
                dists = np.linalg.norm(X_cand_GP[i] - X_cand_GP[chosen], axis=1)
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

        # ensure at least one candidate near incumbent
        fallback_radius = 0.03
        cand_norms = (X_cand_GP - X_cand_GP.min(axis=0)) / (X_cand_GP.max(axis=0) - X_cand_GP.min(axis=0) + 1e-9)
        best_norm = (np.array(x_best_lab[:5], dtype=float) - CONT_MIN) / np.maximum(CONT_RANGE, 1e-9)
        dists_to_best = [np.linalg.norm(cand_norms[idx, :5] - best_norm) for idx in chosen]
        if dists_to_best and np.min(dists_to_best) > fallback_radius:
            jitter = self._local_jitter_points(x_best_lab, count=1)[0]
            if len(chosen) == batch_size:
                worst = int(np.argmin([acq[i] for i in chosen]))
                chosen[worst] = len(X_cand_lab)
            else:
                chosen.append(len(X_cand_lab))
            X_cand_lab.append(jitter)
            X_cand_GP = X_lab_to_GP(X_cand_lab)

        return [X_cand_lab[i] for i in chosen]

    def _current_best(self):
        y_array = np.array(self.Y)
        idx = int(np.argmax(y_array))
        return self.X_lab[idx], float(y_array[idx])

    def _cell_statistics(self):
        stats = {}
        for cell in ['celltype_1', 'celltype_2', 'celltype_3']:
            vals = [y for x, y in zip(self.X_lab, self.Y) if x[5] == cell]
            if vals:
                stats[cell] = {'best': max(vals), 'n': len(vals)}
            else:
                stats[cell] = {'best': float('-inf'), 'n': 0}
        return stats

    def _allowed_cells(self, it):
        base_cells = ['celltype_1', 'celltype_2', 'celltype_3']
        if it < 2:
            return base_cells
        stats = self._cell_statistics()
        ordered = sorted(stats.items(), key=lambda kv: kv[1]['best'], reverse=True)
        available = [cell for cell, meta in ordered if meta['n'] > 0]
        if len(available) <= 1:
            return base_cells
        best_cell, best_meta = ordered[0]
        second_cell, second_meta = ordered[1]
        if best_meta['n'] >= 4 and (best_meta['best'] - second_meta['best']) > 1.0:
            return [best_cell]
        if best_meta['n'] >= 3 and (best_meta['best'] - second_meta['best']) > 0.5:
            return [best_cell, second_cell]
        return available

    def _is_stagnating(self, window=2, tol=0.2):
        if len(self.best_progress) <= window:
            return False
        return (self.best_progress[-1] - self.best_progress[-1 - window]) < tol

    def _local_jitter_points(self, x_best_lab, count=1):
        base = np.array(x_best_lab[:5], dtype=float)
        base_norm = (base - CONT_MIN) / np.maximum(CONT_RANGE, 1e-9)
        jitter_points = []
        for _ in range(count):
            noise = np.random.normal(0.0, 0.02, size=5)
            cont_norm = np.clip(base_norm + noise, 0.0, 1.0)
            cont = CONT_MIN + cont_norm * CONT_RANGE
            jitter_points.append([
                float(cont[0]),
                float(cont[1]),
                float(cont[2]),
                float(cont[3]),
                float(cont[4]),
                x_best_lab[5]
            ])
        return jitter_points


# -----------------------------------------------------
# BO EXECUTION BLOCK (REQUIRED)
# -----------------------------------------------------
BO_m = BO(
    max_iters=10,
    batch_size=5,
    n_init=6,
    n_base=5000,
    multi_hyper=3,
    seed=None,
    time_budget=60
)

# Optional: simple printout for your own debugging (can be removed in final submission)
Y_array = np.array(BO_m.Y)
best_so_far = np.maximum.accumulate(Y_array)
idx_first_best = np.argmax(best_so_far)

print("Best first reached at experiment index (0-based):", idx_first_best)
print("That’s experiment #", idx_first_best + 1, "in total.")
print("Total experiments:", len(BO_m.Y))
print("Max titre found:", max(BO_m.Y))
print("Best X:", BO_m.X_lab[np.argmax(BO_m.Y)])
print("Total runtime (s):", sum(BO_m.time))
