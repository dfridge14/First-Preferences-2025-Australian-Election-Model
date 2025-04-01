import numpy as np
import pandas as pd

def ilr_basis(D):
    """Construct ILR transformation basis matrix for D parts."""
    J = np.zeros((D - 1, D))
    for j in range(D - 1):
        k = j + 1
        coeff = np.sqrt(k / (k + 1))
        J[j, :k] = coeff / k  # Positive part
        J[j, k] = -coeff       # Negative part
    return J

def ilr_transform(x, J):
    """Transform composition to ILR coordinates."""
    if x.ndim == 1:
        x = x.reshape(1, -1)  # Reshape to 2D (1, D) if x is 1D
    return np.dot(x, J.T)  # Matrix multiplication: x @ J.T

def ilr_inverse(z, J):
    """Back-transform ILR to probability space (closure)."""
    """Inverse ILR transformation to recover original proportions."""
    log_x = np.dot(z, J)  # Reverse transform
    x = np.exp(log_x)  # Convert back from log-ratio space

    return x / np.sum(x)

# Step 1: Define the proportions (probabilities) for 4 parties.
proportions = np.array([0.40, 0.35, 0.15, 0.10])

# Step 2: Apply the ILR transformation (using 40% as the reference category).
J = ilr_basis(len(proportions))
ilr_transformed = ilr_transform(proportions, J)


# test of correctness
x = np.array([0.40, 0.35, 0.15, 0.10])
z = ilr_transform(x, J)
x_recovered = ilr_inverse(z, J)

# Step 3: Simulate multinomial data based on the given proportions (1000 samples).
num_samples = 1000  # Total observations for each sample
num_simulations = 500  # Number of different multinomial samples to simulate

# Generate 500 samples of multinomial data, each with 'num_samples' trials
simulated_data = np.array([np.random.multinomial(num_samples, proportions) for _ in range(num_simulations)])
# Step 4: Compute covariance matrix in the original (probability) space
# Convert counts to proportions
simulated_proportions = simulated_data / num_samples

print(simulated_proportions.shape)




# Compute covariance matrix of proportions
cov_matrix_prob_space = np.cov(simulated_proportions.T)

print(cov_matrix_prob_space.shape)


# Step 5: Apply the ILR transformation to the simulated data
ilr_simulated = np.array([ilr_transform(x, J).flatten() for x in simulated_proportions])

print(ilr_simulated)

# Compute covariance matrix in ILR space
cov_matrix_ilr_space = np.cov(ilr_simulated.T)

print(cov_matrix_ilr_space)

# Step 6: Display results
print("Original proportions:", proportions)
print("ILR transformed:", ilr_transformed)
print("\nCovariance matrix in probability space (Original):\n", cov_matrix_prob_space)
print("\nCovariance matrix in ILR space (Transformed):\n", cov_matrix_ilr_space)

# Optional: Back-transform to see recovery of proportions (after covariance estimation)
recovered_proportions = np.array([ilr_inverse(z, J) for z in ilr_simulated])
print("\nRecovered proportions after inverse transformation (mean):", np.mean(recovered_proportions, axis=0))


import pdb;pdb.set_trace()

# Log transformation testing


# Baseline proportion
#p_t = np.array([0.40, 0.35, 0.15, 0.10])
import numpy as np

def compute_log_swings(proportions_old, proportions_new):
    """Compute log swings and center them."""
    log_swings = np.log(proportions_new) - np.log(proportions_old)
    return log_swings - np.mean(log_swings, axis=1, keepdims=True)  # Centering

def estimate_covariance(swings):
    """Estimate the empirical covariance matrix of swings."""
    return np.cov(swings, rowvar=False)

def simulate_log_swings(mean, cov, n_samples):
    """Simulate new log swings from a normal distribution."""
    return np.random.multivariate_normal(mean, cov, size=n_samples)

def backtransform_log_swings(proportions_fresh, log_swings):
    """Convert log swings back to proportions and normalize."""
    new_props = np.exp(np.log(proportions_fresh) + log_swings)
    return new_props / new_props.sum(axis=1, keepdims=True)

# Step 1: Generate Historical Data
n_samples = 1000
categories = 4
proportions_old = np.random.dirichlet([4, 3.5, 1.5, 1], size=n_samples)  # Historical
proportions_new = np.random.dirichlet([4.2, 3.2, 1.4, 1.3], size=n_samples)  # Updated

# Step 2: Compute Log Swings and Estimate Covariance
log_swings = compute_log_swings(proportions_old, proportions_new)
log_mean = np.mean(log_swings, axis=0)
log_cov = estimate_covariance(log_swings)

# Step 3: Simulate New Log Swings
simulated_log_swings = simulate_log_swings(log_mean, log_cov, n_samples)

# Step 4: Apply to Fresh Initial Proportions
proportions_fresh = np.random.dirichlet([5.6, 2.3, 1.5, 0.6], size=n_samples)  # New baseline
simulated_proportions = backtransform_log_swings(proportions_fresh, simulated_log_swings)

# Check Means
print("Mean of original old proportions:", np.mean(proportions_old, axis=0))
print("Mean of original new proportions:", np.mean(proportions_new, axis=0))
print("Mean of original fresh proportions:", np.mean(proportions_fresh, axis=0))
print("Mean of simulated proportions:", np.mean(simulated_proportions, axis=0))

import pdb;pdb.set_trace()


import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

# ---------- Step 1: Generate synthetic compositional data ----------
np.random.seed(42)

num_samples = 100  # Number of historical elections
num_parties = 4    # Number of parties (including reference)

# Generate synthetic predicted and actual vote shares (compositional)
predicted = np.random.dirichlet([3, 2, 4, 1], size=num_samples)  # Simulating predictions
actual = np.random.dirichlet([3, 2, 4, 1], size=num_samples)     # Simulating actual results

# ---------- Step 2: ALR Transformation Function ----------
def alr_transform(comp_data, ref_idx):
    """Applies ALR transformation with specified reference index"""
    non_ref = np.delete(comp_data, ref_idx, axis=1)
    ref_values = comp_data[:, ref_idx].reshape(-1, 1)
    return np.log(non_ref / ref_values)

def alr_inverse(alr_data, ref_idx):
    """Inverse ALR transformation to get back to composition"""
    exp_data = np.exp(alr_data)
    denom = 1 + np.sum(exp_data, axis=1, keepdims=True)
    comp_data = np.insert(exp_data / denom, ref_idx, 1 / denom.flatten(), axis=1)
    return comp_data

# ---------- Step 3: Compute Swings in ALR Space ----------
swings_by_reference = {}
cov_matrices = {}

for ref_idx in range(num_parties):
    alr_pred = alr_transform(predicted, ref_idx)
    alr_act = alr_transform(actual, ref_idx)
    
    swings = alr_act - alr_pred  # Swings in ALR space
    swings_by_reference[ref_idx] = swings
    cov_matrices[ref_idx] = np.cov(swings, rowvar=False)

# ---------- Step 4: Simulate new swings ----------
simulations_by_reference = {}

for ref_idx in range(num_parties):
    mean_swing = np.mean(swings_by_reference[ref_idx], axis=0)
    cov_swing = cov_matrices[ref_idx]

    simulated_swings = multivariate_normal.rvs(mean=mean_swing, cov=cov_swing, size=100)
    simulated_votes = alr_inverse(simulated_swings + alr_transform(predicted, ref_idx), ref_idx)

    simulations_by_reference[ref_idx] = simulated_votes

# ---------- Step 5: Examine Distortions ----------
distortions = {}

for ref_idx in range(num_parties):
    simulated_means = np.mean(simulations_by_reference[ref_idx], axis=0)
    original_means = np.mean(actual, axis=0)

    distortion = simulated_means - original_means
    distortions[ref_idx] = distortion

# ---------- Step 6: Print Results ----------
for ref_idx, dist in distortions.items():
    print(f"Reference Party {ref_idx}: Mean Distortion in Vote Shares")
    print(pd.Series(dist, index=[f"Party {i}" for i in range(num_parties)]))
    print()



import pdb;pdb.set_trace()




import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Generalized normal CDF (using scale α=1, shape β=10)
def generalized_normal_cdf(x, alpha=1, beta=10):
    return stats.norm.cdf(x, loc=0, scale=alpha)

# Inverse Generalized normal CDF
def inverse_generalized_normal_cdf(p, alpha=1, beta=10):
    return stats.norm.ppf(p, loc=0, scale=alpha)

# Constituency proportions
constituency_proportions = np.array([0.52, 0.28, 0.15, 0.05])

# National swing (additive)
national_swing = np.array([0.05, 0.02, -0.01, -0.06])

# Transform constituency proportions using generalized normal CDF
transformed_proportions = generalized_normal_cdf(constituency_proportions)

# Apply the national swing
transformed_proportions_with_swing = transformed_proportions + national_swing

# Ensure no values fall outside [0, 1] range (clipping them to avoid negatives or values > 1)
transformed_proportions_with_swing = np.clip(transformed_proportions_with_swing, 0, 1)

# Inverse transform back to the original space
inverse_transformed_proportions = inverse_generalized_normal_cdf(transformed_proportions_with_swing)

# Renormalize the proportions to sum to 1 (if necessary)
renormalized_proportions = inverse_transformed_proportions / inverse_transformed_proportions.sum()

print("Original Proportions:", constituency_proportions)
print("Transformed Proportions:", transformed_proportions)
print("Transformed with National Swing:", transformed_proportions_with_swing)
print("Renormalized Proportions after Swing:", renormalized_proportions)








import pdb;pdb.set_trace()




import numpy as np
import pandas as pd
from scipy.stats import dirichlet, multivariate_normal

def alr_transform(comp, ref_index=-1):
    """Applies ALR transformation using the given reference category."""
    comp = np.asarray(comp)
    return np.log(comp[:-1] / comp[ref_index])

def inv_alr_transform(alr_vals, ref_val=1):
    """Inverse ALR transformation to get back to composition space."""
    exp_vals = np.exp(alr_vals)
    comp = np.append(exp_vals, ref_val)
    return comp / comp.sum()

# Generate synthetic before-and-after proportions
np.random.seed(42)
before_props = dirichlet.rvs([1, 2, 3, 4], size=100)  # 100 observations
after_props = dirichlet.rvs([0.7, 1.6, 3.2, 4.5], size=100)  # New election

# Compute ALR transforms
alr_before = np.apply_along_axis(alr_transform, 1, before_props)
alr_after = np.apply_along_axis(alr_transform, 1, after_props)

# Compute swings in ALR space
alr_swings = alr_after - alr_before

# Estimate covariance of ALR swings
cov_matrix = np.cov(alr_swings.T)

# Simulate new ALR swings from MVN
simulated_alr_swings = multivariate_normal.rvs(mean=np.mean(alr_swings, axis=0), cov=cov_matrix, size=100)

# Apply simulated swings in ALR space
simulated_alr_after = alr_before + simulated_alr_swings

# Transform back to proportion space
simulated_props_after = np.apply_along_axis(inv_alr_transform, 1, simulated_alr_after)

# Compare original and simulated results
print("Original Swings (Proportion Space):\n", (after_props - before_props)[:3])
print("Simulated Swings (Proportion Space, from ALR):\n", (simulated_props_after - before_props)[:3])



import pdb;pdb.set_trace()

