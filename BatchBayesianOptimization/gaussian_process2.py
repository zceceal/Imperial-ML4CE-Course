import MLCE_CWBO2025.virtual_lab as virtual_lab
from MLCE_CWBO2025.gp_model import GP_model
import sobol_seq
import numpy as np
import random
from datetime import datetime

# Continuous variable bounds used for normalising inputs before training the GP.
CONT_MIN = np.array([30.0, 6.0, 0.0, 0.0, 0.0])
CONT_MAX = np.array([40.0, 8.0, 50.0, 50.0, 50.0])
CONT_RANGE = CONT_MAX - CONT_MIN

"""
gp_helpers.py

Helper functions for:
- encoding the cell type,
- converting LAB-format inputs to GP-format,
- calling the virtual lab,
- generating the initial design,
- (optionally) building and testing a GP.

This file is imported by the BO script.
"""


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
        cont = np.array(row[:5], dtype=float)  # T, pH, F1, F2, F3
        cont_norm = (cont - CONT_MIN) / np.maximum(CONT_RANGE, 1e-12)
        cell_str = row[5]          # celltype string
        cell_oh = encode_cell_type(cell_str)
        X_num.append(list(cont_norm) + cell_oh)

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


# -----------------------------------------------------------
# 6) Small test when running this file directly
# -----------------------------------------------------------

if __name__ == "__main__":
    # Simple test: fit GP on 6 initial points and predict at one new point
    Xtrain_lab, Ytrain, Xtrain_GP, GP_m = build_trained_gp(n_init=6, multi_hyper=1)

    x_new_lab = [35.0, 7.0, 25.0, 25.0, 25.0, 'celltype_1']
    X_new_GP = X_lab_to_GP([x_new_lab])   # shape (1, 8)
    x_new_vec = X_new_GP[0]               # shape (8,)

    mean_vec, var_vec = GP_m.GP_inference_np(x_new_vec)
    mu_pred = float(mean_vec[0])
    var_pred = float(var_vec[0])

    print("Test GP:")
    print("  Predicted mean titre:", mu_pred)
    print("  Predicted variance  :", var_pred)
