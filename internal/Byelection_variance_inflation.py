

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os
os.chdir('/home/dania-freidgeim/Australian Election')


X_by_seat  = pd.read_csv('X_by_seat.csv').set_index('Electorate')   # full ALR residuals (BY-SEAT, 3D)
X_by_seat.loc[:, 'n'] = np.array([3,3,2,3,2,3,3,3,3])


BY_SEAT_partial = [
    {
        "p_poll": np.array([0.547, 0.228, 0.225]),
        "p_result": np.array([0.482, 0.157, 0.361]),
        "parties": ["COAL", "GRN", "OTH"],"n":1
    },
    {
        "p_poll": np.array([0.413, 0.402, 0.185]),
        "p_result": np.array([0.431, 0.395, 0.174]),
        "parties": ["ALP", "GRN", "OTH"],"n":1
    },
    {
        "p_poll": np.array([0.511, 0.166, 0.322]),
        "p_result": np.array([0.526, 0.165, 0.309]),
        "parties": ["ALP", "GRN", "OTH"],"n":1
    },
    {
        "p_poll": np.array([0.475, 0.18, 0.346]),
        "p_result": np.array([0.393, 0.188, 0.419]),
        "parties": ["ALP", "GRN", "OTH"],"n":1
    },
    {
        "p_poll": np.array([0.601, 0.164, 0.235]),
        "p_result": np.array([0.627, 0.165, 0.209]),
        "parties": ["COAL", "GRN", "OTH"],"n":1
    }
]


SEAT_POLL_SCALE_1 = 1.0747964937804577
SEAT_POLL_SCALE_2 = 0.8977871632158285
SEAT_POLL_SCALE_3 = 0.7965537354979125

# ==========================================
# 1. FIT THE VARIANCE DECAY CURVE
# ==========================================

# Using the scales from your general election buckets
n_vals = np.array([1, 2, 3.5]) # Average n for the 3+ bucket is ~3.5
var_ratios = np.array([SEAT_POLL_SCALE_1**2, SEAT_POLL_SCALE_2**2, SEAT_POLL_SCALE_3**2])

def var_decay_model(n, A, B):
    return A + B / n

params, _ = curve_fit(var_decay_model, n_vals, var_ratios)
A, B = params

# Lock in the 1.54 baseline
v_baseline = var_decay_model(1.54, A, B)

def get_sigma_scalars(n_array):
    """
    Returns an array of covariance multipliers, capped at n=3.
    Expects n_array to be a numpy array or pandas Series.
    """
    n_capped = np.clip(n_array, a_min=1, a_max=3)
    return var_decay_model(n_capped, A, B) / v_baseline

# Optional: Print out the scalars to verify the 1.54 landing spot
print(f"Scalar at n=1.00: {get_sigma_scalars(np.array([1]))[0]:.3f}")
print(f"Scalar at n=1.54: {get_sigma_scalars(np.array([1.54]))[0]:.3f} (Baseline)")
print(f"Scalar at n=2.00: {get_sigma_scalars(np.array([2]))[0]:.3f}")
print(f"Scalar at n=3.00: {get_sigma_scalars(np.array([3]))[0]:.3f} (Cap)")


# ==========================================
# 2. VECTORIZED FULL SEAT ENERGY (X_by_seat)
# ==========================================

# Assumes X_by_seat has an 'n' column and the ALR dimensions
# X_by_seat = pd.read_csv('X_by_seat.csv').set_index('Electorate')



# Extract n values and calculate scalars
n_seat_full = X_by_seat['n'].values
scalars_full = get_sigma_scalars(n_seat_full)

# Extract just the ALR residual columns (dropping 'n')
# Adjust column names if your ALR columns differ
X_alr_only = X_by_seat.drop(columns=['n']).values 

# Base Sigma_gen_seat and its inverse
X_gen_seat = pd.read_csv('X_gen_seat.csv').iloc[:,:5].set_index(['Election_year','Electorate'])    # GEN baseline residuals (optional diagnostics only)
Sigma_gen_seat = np.cov(X_gen_seat, rowvar=False)
Sigma_seat_inv = np.linalg.inv(Sigma_gen_seat)

# Calculate baseline energy using np.einsum
baseline_energies_full = np.einsum('ij,jk,ik->i', X_alr_only, Sigma_seat_inv, X_alr_only)

# Divide by the scalar array to get intensity-adjusted energy per seat
# (Since Sigma_n = c * Sigma_base, Sigma_n_inv = (1/c) * Sigma_base_inv)
E_seat_full_weighted = np.sum(baseline_energies_full / scalars_full)


# ==========================================
# 3. ITERATIVE PARTIAL SEAT ENERGY
# ==========================================

def to_alr(p, ref_idx):
    p = np.asarray(p, dtype=float)
    keep = [i for i in range(len(p)) if i != ref_idx]
    return np.log(p[keep] / p[ref_idx])

def process_partial_weighted(partial_list, Sigma_base):
    total_energy = 0.0
    
    for obs in partial_list:
        # Extract n, compute local scalar
        n_obs = obs["n"]
        scalar_obs = get_sigma_scalars(np.array([n_obs]))[0]
        
        # Scale the base matrix for THIS observation's intensity
        Sigma_n = Sigma_base * scalar_obs
        
        p_poll = obs["p_poll"]
        p_res  = obs["p_result"]
        parties = obs["parties"]
        
        ref = parties.index("COAL") if "COAL" in parties else 0
        y = to_alr(p_poll, ref) - to_alr(p_res, ref)
        
        keep = [i for i in range(3) if i != ref]
        A_proj = np.zeros((len(y), 3))
        A_proj[:, keep] = np.eye(len(y))
        
        # Project the heavily/lightly polled matrix into 2D space
        S_local = A_proj @ Sigma_n @ A_proj.T
        total_energy += y.T @ np.linalg.solve(S_local, y)
        
    return total_energy

E_seat_partial_weighted = process_partial_weighted(BY_SEAT_partial, Sigma_gen_seat)


# ==========================================
# 4. FINAL CALCULATION
# ==========================================

# Total degrees of freedom: 3 per full observation, 2 per partial
df_seat = (3 * len(X_by_seat)) + (2 * len(BY_SEAT_partial))

# Calculate rigorous s_seat
s_seat_rigorous = (E_seat_full_weighted + E_seat_partial_weighted) / df_seat

print(f"\nRigorous s_seat (Intensity-Adjusted): {s_seat_rigorous:.4f}")
print(f"Implied Standard Deviation Premium:   {np.sqrt(s_seat_rigorous):.2%}")







import pdb; pdb.set_trace()
# previous code!

# 1. get residuals data
os.chdir('/home/dania-freidgeim/Australian Election')
X_by_nat = pd.read_csv('X_by_nat.csv').set_index('div_nm')     # full ALR residuals (BY-NAT, 3D where available)
X_by_seat  = pd.read_csv('X_by_seat.csv').set_index('Electorate')   # full ALR residuals (BY-SEAT, 3D)
X_gen_seat = pd.read_csv('X_gen_seat.csv').iloc[:,:5].set_index(['Election_year','Electorate'])    # GEN baseline residuals (optional diagnostics only)
Sigma_gen_seat = np.cov(X_gen_seat, rowvar=False)
Sigma_gen_nat = np.array([
    [0.058233, 0.026137, 0.018969],
    [0.026137, 0.076755, 0.011659],
    [0.018969, 0.011659, 0.150876]
]) + np.array([
    [0.009037, 0.002452, 0.003341],
    [0.002452, 0.027528, 0.012779],
    [0.003341, 0.012779, 0.034484]
]) # Σ_GEN-NAT (already Σ_poll + Σ_residual)


# partial data in PROPORTION space:
BY_NAT_partial = []  # list of dicts: {"p": vec, "ref": str}

BY_SEAT_partial = [
    {
        "p_poll": np.array([0.547, 0.228, 0.225]),
        "p_result": np.array([0.482, 0.157, 0.361]),
        "parties": ["COAL", "GRN", "OTH"],"n":1
    },
    {
        "p_poll": np.array([0.413, 0.402, 0.185]),
        "p_result": np.array([0.431, 0.395, 0.174]),
        "parties": ["ALP", "GRN", "OTH"],"n":1
    },
    {
        "p_poll": np.array([0.511, 0.166, 0.322]),
        "p_result": np.array([0.526, 0.165, 0.309]),
        "parties": ["ALP", "GRN", "OTH"],"n":1
    },
    {
        "p_poll": np.array([0.475, 0.18, 0.346]),
        "p_result": np.array([0.393, 0.188, 0.419]),
        "parties": ["ALP", "GRN", "OTH"],"n":1
    },
    {
        "p_poll": np.array([0.601, 0.164, 0.235]),
        "p_result": np.array([0.627, 0.165, 0.209]),
        "parties": ["COAL", "GRN", "OTH"],"n":1
    }
]

PARTIES = ['COAL','ALP', 'GRN']  # adapt if needed


# 2. GRN bias modelling - correct/shrink to get bias relative to non-COAL ALR parties' bias? +0.095 - modelling choice - overall bias in sample driven by COAL


mu_grn = X_by_nat.mean()['GRN'] - (X_by_nat.mean()['ALP'] + X_by_nat.mean()['OTH'])/2 # equivalent to shrinkage with kappa = 4

X_by_nat['GRN']  = X_by_nat['GRN'] - mu_grn

# 3. Invert CovMs
Sigma_nat_inv = np.linalg.inv(Sigma_gen_nat)
Sigma_seat_inv = np.linalg.inv(Sigma_gen_seat)

# 4. Estimate energy
def mahalanobis_energy_winsorised(X, Sigma, cap=10.0):
    """
    Winsorised Mahalanobis energy:
    caps per-observation quadratic contributions.
    """
    Sinv = np.linalg.inv(Sigma)
    E = 0.0

    for x in X:
        e = x.T @ Sinv @ x
        E += min(e, cap)
    return E

E_nat_full = mahalanobis_energy_winsorised(X_by_nat.values, Sigma_gen_nat, cap=10.0) # limit impact of New England and Mayo (un-adjusted IND/XENs)
E_seat_full = np.einsum('ij,jk,ik->i',X_by_seat, Sigma_seat_inv, X_by_seat).sum() # seat polls takes as they are

# Contributions of partial alr data - missing parties

def to_alr(p, ref_idx):
    p = np.asarray(p, dtype=float)
    keep = [i for i in range(len(p)) if i != ref_idx]
    return np.log(p[keep] / p[ref_idx])

def process_partial(partial_list, Sigma):
    energies = []

    for obs in partial_list:
        p_poll = obs["p_poll"]
        p_res  = obs["p_result"]
        parties = obs["parties"]

        # choose reference (COAL if present, else first party)
        if "COAL" in parties:
            ref = parties.index("COAL")
        else:
            ref = 0

        # ALR residual = log(poll) - log(result)
        y_poll = to_alr(p_poll, ref)
        y_res  = to_alr(p_res, ref)
        y = y_poll - y_res

        # build local projection matrix A (maps 3D ALR → 2D ALR)
        keep = [i for i in range(3) if i != ref]

        A = np.zeros((2, 3))
        A[:, keep] = np.eye(2)

        S = A @ Sigma @ A.T

        energies.append(y.T @ np.linalg.solve(S, y))

    return sum(energies)

E_nat_partial = process_partial(BY_NAT_partial, Sigma_gen_nat) # I think correct Sigmas here
E_seat_partial = process_partial(BY_SEAT_partial, Sigma_gen_seat)
E_total = (E_nat_full +E_seat_full +E_nat_partial +E_seat_partial)


df_total = (
    3 * (len(X_by_nat) + len(X_by_seat)) +
    2 * (len(BY_NAT_partial) + len(BY_SEAT_partial))
)
s = E_total / df_total

s_nat = E_nat_full / (3 * len(X_by_nat))
s_seat = (E_seat_full+E_seat_partial) / (3 * len(X_by_seat) + 2 * (len(BY_SEAT_partial)))

print('s:', s)
import pdb; pdb.set_trace()