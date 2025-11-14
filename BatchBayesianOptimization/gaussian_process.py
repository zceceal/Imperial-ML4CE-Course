import MLCE_CWBO2025.virtual_lab as virtual_lab
from MLCE_CWBO2025.gp_model import GP_model
import sobol_seq
import numpy as np
import random
from datetime import datetime


def generate_initial_design_lab(n_init=6, seed=None):
    """
    Generate n_init initial points in LAB format:
    [T, pH, F1, F2, F3, cell_type_str]
    using a Sobol sequence for the 5 continuous variables,
    and balanced assignment of the 3 cell types.
    """

    if seed is not None:
        np.random.seed(seed)

    # 1. Sobol in [0,1]^5
    sobol_points = sobol_seq.i4_sobol_generate(5, n_init)  # shape (n_init, 5)

    # 2. Scale to real ranges
    T_vals  = 30 + sobol_points[:, 0] * (40 - 30)
    pH_vals = 6  + sobol_points[:, 1] * (8  - 6)
    F1_vals = 0  + sobol_points[:, 2] * 50
    F2_vals = 0  + sobol_points[:, 3] * 50
    F3_vals = 0  + sobol_points[:, 4] * 50

    # 3. Cell types: ensure we cover all 3 fairly
    base_cells = ['celltype_1', 'celltype_2', 'celltype_3']
    cell_types = (base_cells * (n_init // 3 + 1))[:n_init]  # repeat & cut
    # Optionally shuffle the order:
    cell_types = np.array(cell_types)
    np.random.shuffle(cell_types)

    # 4. Combine into lab-format list-of-lists
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

def encode_cell_type(cell_str):
    """Convert 'celltype_1/2/3' → one-hot [c1, c2, c3]."""
    if cell_str == 'celltype_1':
        return [1.0, 0.0, 0.0]
    elif cell_str == 'celltype_2':
        return [0.0, 1.0, 0.0]
    elif cell_str == 'celltype_3':
        return [0.0, 0.0, 1.0]
    else:
        raise ValueError(f"Unknown cell type: {cell_str}")

def X_lab_to_GP(X_lab):
    """
    Convert lab-format X (with string cell types) to numeric GP-format X.

    X_lab: list or array of rows [T, pH, F1, F2, F3, cell_type_str]
    Returns: np.array of shape (N, 8)
    """
    X_num = []
    for row in X_lab:
        # row is something like [T, pH, F1, F2, F3, 'celltype_1']
        cont = row[:5]                         # first 5 continuous features
        cell_str = row[5]
        cell_oh = encode_cell_type(cell_str)   # 3-dim one-hot
        X_num.append(list(cont) + cell_oh)     # 5 + 3 = 8 dims total

    return np.array(X_num, dtype=float)

def objective_func(X_lab):
    # X_lab is list/array of rows [T, pH, F1, F2, F3, cell_type_str]
    return np.array(virtual_lab.conduct_experiment(X_lab)).reshape(-1, 1)

Xtrain_lab = generate_initial_design_lab(n_init=6, seed=42)
Ytrain     = objective_func(Xtrain_lab)
Xtrain_GP  = X_lab_to_GP(Xtrain_lab)

GP_m = GP_model(Xtrain_GP, Ytrain, kernel='RBF', multi_hyper=3, var_out=True)

