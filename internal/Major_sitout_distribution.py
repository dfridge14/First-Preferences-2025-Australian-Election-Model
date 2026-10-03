import pandas as pd
import numpy as np
import os

# =========================
# 1. LOAD DATA
# =========================

os.chdir("/home/dania-freidgeim/Australian Election")
df = pd.read_csv("Major_sitouts_redistribution.csv").iloc[:15,]

# Ensure missing values are zero (important for IND columns)
df = df.fillna(0)

# =========================
# 2. COMPUTE GAINS
# =========================
df['aligned_gain'] = df['aligned_result'] - df['aligned_last_adj']
df['ind_gain'] = df['IND_result'] - df['IND_last_adj']
df['other_major_gain'] = df['other_major_result'] - df['other_major_last_adj']
df['other_gain'] = df['other_result'] - df['other_adj']

# =========================
# 3. KEEP ONLY POSITIVE GAINS
# =========================
gain_cols = ['aligned_gain', 'ind_gain', 'other_major_gain', 'other_gain']
df[gain_cols] = df[gain_cols].clip(lower=0)

# =========================
# 4. TOTAL GAIN
# =========================
df['total_gain'] = df[gain_cols].sum(axis=1)

# =========================
# 5. HANDLE EDGE CASES
# =========================
# Avoid divide-by-zero
df.loc[df['total_gain'] == 0, 'total_gain'] = np.nan

# =========================
# 6. COMPUTE THETA
# =========================
df['theta_aligned'] = df['aligned_gain'] / df['total_gain']
df['theta_ind'] = df['ind_gain'] / df['total_gain']
df['theta_other_major'] = df['other_major_gain'] / df['total_gain']
df['theta_other'] = df['other_gain'] / df['total_gain']


# =========================
# 7. HANDLE NO-IND CASES
# =========================
no_ind_mask = df['has_strong_IND'] == 0

# Set ind to 0
df.loc[no_ind_mask, 'theta_ind'] = 0

# Renormalise remaining
renorm_cols = ['theta_aligned', 'theta_other_major', 'theta_other']

row_sums = df.loc[no_ind_mask, renorm_cols].sum(axis=1)

df.loc[no_ind_mask, renorm_cols] = df.loc[no_ind_mask, renorm_cols].div(row_sums, axis=0)

# =========================
# 8. CLEAN OUTPUT
# =========================
theta_cols = ['theta_aligned', 'theta_ind', 'theta_other_major', 'theta_other']

df_theta = df[['div_nm', 'byelection_year', 'Major_sitout', 'has_strong_IND'] + theta_cols]

# =========================
# 9. GROUPED ESTIMATES
# =========================
grouped = df_theta.groupby(['Major_sitout', 'has_strong_IND'])[theta_cols].mean()

print("\nGrouped theta estimates:\n")
print(grouped)

# =========================
# 10. ADD UNCERTAINTY (DIRICHLET PARAMS)
# =========================

theta_array=df_theta.groupby(['Major_sitout','has_strong_IND'])[theta_cols].apply(lambda d:d.values)

def alpha0_from_group(X):
    m=X.mean(axis=0)
    v=X.var(axis=0,ddof=1)
    mask=m>0.05   # ignore tiny components
    return np.maximum(np.nanmean(m[mask]*(1-m[mask])/(v[mask]+1e-9)-1),1e-3)

alpha0=theta_array.apply(alpha0_from_group)
alpha=grouped.mul(alpha0,axis=0)
print(alpha0)

from scipy.special import psi,polygamma
import numpy as np

def get_support(df):
    if df['has_strong_IND'].iloc[0]==0:
        return ['theta_aligned','theta_other_major','theta_other']
    else:
        return ['theta_aligned','theta_ind','theta_other_major','theta_other']

import numpy as np
from scipy.special import psi

import numpy as np
from scipy.special import psi

def estimate_alpha0_penalised(X, lam=1.0, iters=100):
    X = X / X.sum(axis=1, keepdims=True)
    logX = np.log(X + 1e-12).mean(axis=0)

    k = X.shape[1]
    alpha = np.ones(k)

    for _ in range(iters):
        alpha0 = alpha.sum()

        # gradient of log-likelihood
        grad = psi(alpha0) - psi(alpha) + logX

        # shrinkage toward reasonable scale
        penalty = lam * (alpha - 5.0 / k)

        alpha = np.maximum(alpha + 0.05 * (grad - penalty), 1e-6)

    return alpha.sum()

X = df_theta[theta_cols].values
alpha0 = estimate_alpha0_penalised(X)
print(alpha0)

import pdb; pdb.set_trace()
