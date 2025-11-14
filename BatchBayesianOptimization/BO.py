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


# -----------------------------------------------------
# HELPER 1: generate initial Sobol design
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

    base_cells = ['celltype_1', 'celltype_2', 'celltype_3']
    cell_types = (base_cells * (n_init // 3 + 1))[:n_init]
    cell_types = np.array(cell_types)
    np.random.shuffle(cell_types)

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
# HELPER 2: encode categorical variable
# -----------------------------------------------------
def encode_cell_type(cell_str):
    if cell_str == 'celltype_1':
        return [1,0,0]
    if cell_str == 'celltype_2':
        return [0,1,0]
    if cell_str == 'celltype_3':
        return [0,0,1]
    raise ValueError("Unknown cell type")


# -----------------------------------------------------
# HELPER 3: convert LAB format -> GP numeric format
# -----------------------------------------------------
def X_lab_to_GP(X_lab):
    X_num = []
    for row in X_lab:
        cont = row[:5]
        oh   = encode_cell_type(row[5])
        X_num.append(list(cont) + oh)
    return np.array(X_num, float)


# -----------------------------------------------------
# HELPER 4: wrapper for virtual lab
# -----------------------------------------------------
def objective_func(X_lab):
    return np.array(virtual_lab.conduct_experiment(X_lab)).reshape(-1,1)


# -----------------------------------------------------
# HELPER 5: acquisition function (UCB)
# -----------------------------------------------------
def acquisition_ucb(X_cand_GP, gp, kappa=2.0):
    N = X_cand_GP.shape[0]
    acq = np.zeros(N)
    for i in range(N):
        m, v = gp.GP_inference_np(X_cand_GP[i])
        acq[i] = float(m[0]) + kappa*np.sqrt(max(v[0],0))
    return acq


# -----------------------------------------------------
# HELPER 6: Candidate generator for BO
# -----------------------------------------------------
def generate_candidate_batch_lab(n_cand=200):
    sobol_points = sobol_seq.i4_sobol_generate(5, n_cand)

    T  = 30 + sobol_points[:,0]*10
    pH = 6  + sobol_points[:,1]*2
    F1 = sobol_points[:,2]*50
    F2 = sobol_points[:,3]*50
    F3 = sobol_points[:,4]*50

    cell_types = np.random.choice(['celltype_1','celltype_2','celltype_3'], size=n_cand)

    X = []
    for i in range(n_cand):
        X.append([float(T[i]), float(pH[i]), float(F1[i]), float(F2[i]), float(F3[i]), cell_types[i]])
    return X


# -----------------------------------------------------
# BO CLASS
# -----------------------------------------------------
class BO:
    def __init__(self,
                 max_iters=15,
                 batch_size=5,
                 n_init=6,
                 n_cand=200,
                 multi_hyper=3,
                 seed=None):

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

        # ---------- INITIAL DESIGN ----------
        X_init = generate_initial_design_lab(n_init=n_init, seed=seed)
        Y_init = objective_func(X_init)

        self.X_lab = list(X_init)
        self.Y     = Y_init.flatten().tolist()

        # fit initial GP
        X_GP = X_lab_to_GP(self.X_lab)
        self.gp = GP_model(X_GP, Y_init, 'RBF', multi_hyper, True)

        # record time
        elapsed = datetime.timestamp(datetime.now()) - start_time
        self.time += [elapsed] + [0]*(n_init-1)
        start_time = datetime.timestamp(datetime.now())

        # ---------- BO LOOP ----------
        for it in range(max_iters):

            if sum(self.time) > 60:
                break

            # propose new batch
            X_batch = self._propose_batch(n_cand, batch_size)

            # evaluate
            Y_batch = objective_func(X_batch).flatten().tolist()

            # store
            self.X_lab += X_batch
            self.Y     += Y_batch

            # refit GP
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

        acq = acquisition_ucb(X_cand_GP, self.gp, kappa=2.0)
        idx = np.argsort(-acq)[:batch_size]

        return [X_cand_lab[i] for i in idx]


# -----------------------------------------------------
# BO EXECUTION BLOCK (REQUIRED)
# -----------------------------------------------------
BO_m = BO(
    max_iters=15,
    batch_size=5,
    n_init=6,
    n_cand=200,
    multi_hyper=3,
    seed=42
)

print("Final number of points:", len(BO_m.Y))
print("Total time:", sum(BO_m.time))
print("Best Y:", max(BO_m.Y))
print("Best X:", BO_m.X_lab[np.argmax(BO_m.Y)])