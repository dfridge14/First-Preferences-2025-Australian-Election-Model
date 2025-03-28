import numpy as np
import pandas as pd
import os
import arviz as az
from datetime import datetime
from matplotlib import pyplot as plt
from pathlib import Path


from scipy.linalg import cholesky
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

#pt.config.mode = 'NUMBA'


base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)

election_year = '2016'


#import pdb;pdb.set_trace()

# Assuming df_polling is your dataframe
# We will use the polling data for the 5 electorates
#polling_data = Poll_Swings_National.values  # Convert DataFrame to NumPy array (2D)
#1 = (df_polling.index - df_polling.index[0]).days  # Convert 1 to 'days since start'



n_parties = 3
num_polling_days = 30

# Bayesian Multivariate Gaussian Random Walk for Polling Data
# Using Stan and Python (CmdStanPy for efficiency)

import numpy as np
import pandas as pd
import cmdstanpy
import matplotlib.pyplot as plt
import arviz as az

election_year = '2016'
election_date_num = {'2016':1028}

# Step 1: Load Data (Placeholder, replace with actual data)
# Columns: [day, sample_size, party_1, party_2, party_3, party_4]
df = pd.read_csv(f'NationalPollsforMGRW{election_year}.csv')

# only do inference for last 100 days of election:
starting_point = election_date_num[election_year] - num_polling_days # start 100 days before last day of polling
df = df.loc[df['Days since last election'] >= starting_point,]
df.loc[:,'Days since last election'] -= starting_point
df = df.rename(columns={'Days since last election':'Day_index'})

#import pdb;pdb.set_trace()


# Step 2: Choose Reference Category (e.g., the largest average party)
ref_party = 'ALP'
ref_col_index = 1

# Step 3: ALR Transformation (remove reference party)
def alr_transform(vote_shares, ref_col):
    alr_values = np.log(vote_shares.drop(columns=[ref_col]).values) - np.log(vote_shares[ref_col].values[:, None])
    return alr_values

original_proportions = df.iloc[:, 2:].to_numpy()
alr_votes = alr_transform(df.iloc[:, 2:], ref_party)
days = df['Day_index'].values
sample_sizes = df['Sample size'].values



# Step 4: Infer Poll Variance (multinomial-derived covariance)
def poll_covariance(proportions, sample_size):
    k = len(proportions)
    cov_matrix = np.diag(proportions) - np.outer(proportions, proportions)
    return cov_matrix / sample_size

poll_covs = np.array([poll_covariance(row, n) for row, n in zip(df.iloc[:, 2:].values, sample_sizes)])

#import pdb;pdb.set_trace()

observed_days = np.array(sorted(set(days))) 
N_obs = len(days)
day_indices = np.array([np.where(observed_days == d)[0][0] + 1 for d in days]) 

y_obs = np.array(alr_votes)  # No need for additional filtering
Sigma_poll_obs = np.array(poll_covs)


# transform Sigma_poll_obs to ALR
Sigma_poll_obs_alr = np.zeros((Sigma_poll_obs.shape[0], n_parties, n_parties))

for i, cov in enumerate(Sigma_poll_obs):
    # Extract the relevant part of the covariance matrix (original 4x4)
    cov_original = cov

    # Get the proportions for this observation (assuming columns: LNP, ALP, Greens, Others)
    p = original_proportions[i]  # This is a row of proportions, e.g., [LNP, ALP, Greens, Others]

    # ALR Transformation: use ALP (second column) as reference category
    # ALR1 = log(LNP / ALP)
    # ALR2 = log(Greens / ALP)
    # Others are dropped in the ALR transformation

    # Jacobian matrix for the ALR transformation (3x3)
    # The Jacobian for ALR when ALP is the reference category (2nd column)
    # J is a 3x4 matrix where each row corresponds to the partial derivatives for the ALR dimensions
    if election_year == '2016':
        J = np.array([
        [1/p[0], -1/p[1], 0, 0],
        [0, -1/p[1], 1/p[2], 0],
        [0, -1/p[1], 0, 1/p[3]]
        ])
    
    # Calculate the transformed covariance matrix (3x3 ALR covariance)

    # Store the transformed covariance matrix in Sigma_poll_obs_alr
    Sigma_poll_obs_alr[i] = J @ cov_original @ J.T  # Matrix multiplication for ALR transformation


#import pdb;pdb.set_trace()

# Step 5: Stan Model Definition (Multivariate Gaussian RW with poll uncertainty)
stan_code = """
data {
    int<lower=1> T;  // Total time steps (e.g., 1027)
    int<lower=1> N_obs;  // Total number of observations (e.g., 249)
    int<lower=1> K;  // Number of ALR dimensions (parties minus one)
    array[N_obs] int<lower=1, upper=T> day;  // Observation days

    matrix[N_obs, K] y_obs;  // Observed ALR vote shares
    array[N_obs] matrix[K, K] Sigma_poll;  // Polling uncertainty covariance matrices
}

parameters {
    matrix[T, K] x;  // Latent ALR vote shares
    vector[K] mu;  // Mean drift (trend)

    vector<lower=0>[K] sigma_rw;  // Scale of random walk
}

transformed parameters {
    matrix[K, K] Sigma_rw;  // Covariance matrix for random walk

    // Diagonal prior for the covariance matrix Sigma_rw (independent dimensions)
    Sigma_rw = diag_matrix(to_vector([sigma_rw[1], sigma_rw[2], sigma_rw[3]]));
}


model {
    // Priors
    mu ~ normal(0, 0.1);
    sigma_rw ~ normal(0, 0.1); // Weakly informative scale prior
    
    // Random Walk Process
    for (t in 2:T) {
        real dt = 1.0;  // Default step is 1 day
        if (t > 1) {
            dt = t - (t - 1);  // Handle non-uniform spacing
        }
        x[t] ~ multi_normal(to_vector(x[t-1]) + dt * mu, dt * Sigma_rw);
    }

    // Observations: Combine multiple polls per day
    for (t in 1:T) {
        int count = 0;
        vector[K] weighted_sum_y = rep_vector(0, K);
        matrix[K, K] precision_sum = rep_matrix(0, K, K);
        
        for (j in 1:N_obs) {
            if (day[j] == t) {  // Collect polls for day t
                precision_sum += inverse(Sigma_poll[j]);  // Sum of precision matrices
                weighted_sum_y += inverse(Sigma_poll[j]) * to_vector(y_obs[j]);  // Precision-weighted sum
                count += 1;
            }
        }
        
        if (count > 0) {
            matrix[K, K] Sigma_combined = inverse(precision_sum);
            vector[K] y_combined = Sigma_combined * weighted_sum_y;
            x[t] ~ multi_normal(y_combined, Sigma_combined);
        }
    }
}
"""

stan_filename = "polling_mgrw.stan"
with open(stan_filename, "w") as f:
    f.write(stan_code)

# Step 6: Compile & Fit Stan Model
stan_model = cmdstanpy.CmdStanModel(stan_file=stan_filename)
data_dict = {
    'T': max(days)+1,
    'K': alr_votes.shape[1],
    'N_obs': N_obs,
    'day': day_indices,
    'y_obs': y_obs,
    'Sigma_poll': Sigma_poll_obs_alr
}
fit = stan_model.sample(data=data_dict, chains=4, parallel_chains=4, iter_warmup=500, iter_sampling=1000)

import pdb;pdb.set_trace()


# Step 7: Extract & Plot Results
idata = az.from_cmdstanpy(posterior=fit)

x_samples = idata.posterior['x'].values  # Shape: (chains, draws, T, K)

# Reshape to merge chains and draws
x_samples = x_samples.reshape(-1, x_samples.shape[2], x_samples.shape[3])  # Shape: (total_samples, T, K)

for i in range(x_samples.shape[2]):  # Iterate over K dimensions (ALR vote shares)
    az.plot_hdi(np.arange(x_samples.shape[1]), x_samples[:, :, i], color='blue', fill_kwargs={'alpha': 0.3})  
    plt.plot(np.arange(x_samples.shape[1]), np.median(x_samples[:, :, i], axis=0), label=f'Party {i+1}')

plt.xlabel('Days Since Election')
plt.ylabel('ALR Vote Share')
plt.legend()
plt.show()

import pdb;pdb.set_trace()

def alr_inverse(alr_samples):
    """Convert ALR-transformed data back to simplex space (vote shares)."""
    print('done')
    exp_values = np.exp(alr_samples)  # Exponentiate all ALR values
    last_column = 1 / (1 + np.sum(exp_values, axis=-1, keepdims=True))  # Compute last component
    return np.concatenate([exp_values * last_column, last_column], axis=-1)  # Reconstruct simplex

x_samples_simplex = np.zeros((4000, num_polling_days, n_parties+1))  # Same shape as x_samples + 1 col!

# Iterate over the 4000 samples (first dimension)
for i in range(x_samples.shape[0]):
    x_samples_simplex[i] = alr_inverse(x_samples[i]) 

# Plot the back-transformed vote shares
plt.figure(figsize=(10, 5))
for i in range(x_samples_simplex.shape[2]):  # Iterate over parties (including the reference category)
    az.plot_hdi(np.arange(x_samples_simplex.shape[1]), x_samples_simplex[:, :, i], color='blue', fill_kwargs={'alpha': 0.3})
    plt.plot(np.arange(x_samples_simplex.shape[1]), np.median(x_samples_simplex[:, :, i], axis=0), label=f'Party {i+1}')

plt.xlabel('Days Since Election')
plt.ylabel('Vote Share')
plt.legend()
plt.show()

import pdb;pdb.set_trace()

posterior_samples = fit.stan_variable("mu")  # For mean vector (mu)
cov_matrix_samples = fit.stan_variable("cov_matrix")
    

def old_model_PyMC():
    import pymc as pm
    import pytensor.tensor as pt


    Poll_Swings_National = pd.read_csv(f"NationalPollsforMGRW{election_year}.csv", index_col = None)

    Y = Poll_Swings_National.iloc[:,2].values
    sample_size = Poll_Swings_National['Sample size'].values
    time_idx = Poll_Swings_National['Days since last election'].values

    # Create a PyMC3 model
    if __name__ == '__main__':
        with pm.Model() as model:
            # Priors for the mean and covariance of the random walk
            ###mu = pm.Normal('mu', mu=0, sigma=1, shape=(5,))
            ###cov_matrix = pm.LKJCholeskyCov('cov_matrix', n=5, eta=2., sd_dist=pm.HalfNormal.dist(1), shape=(5, 5))
            
            # Cholesky decomposition of the covariance matrix
            ###L = pm.expand_packed(L=pm.math.cholesky(cov_matrix))
            # Prior for initial state (party swings at t=0)

            # The random walk over time
            #random_walk = pm.MvNormal('random_walk', mu=mu, cov=L, shape=(len(dates), 5), observed=polling_data)
            
            # Sampling
            #trace = pm.sample(1000, return_inferencedata=False)
            
        
            #import pdb;pdb.set_trace()

            # Multivariate Gaussian Random Walk
            #sigma_rw = pm.HalfNormal("sigma_rw", sigma=0.02, shape=n_parties)  
            #sigma_rw = pt.eye(n_parties) * 0.02 # Random walk step size

            sd_dist = pm.HalfNormal.dist(sigma=0.01, shape=n_parties)
            #sigma_chol, _, _ = pm.LKJCholeskyCov("sigma_rw", n=n_parties, eta=2, sd_dist=sd_dist)  # Unpack correctly
            #sigma_rw = pm.math.dot(sigma_chol, sigma_chol.T)  # Convert Cholesky factor to full covariance matrix    #sigma_rw = pm.expand_packed_triangular(n_parties, sigma_chol, lower=True)


            chol_packed = pm.LKJCholeskyCov("chol_sigma_rw", n=n_parties, eta=2, sd_dist=sd_dist, compute_corr = False)
            chol = pm.expand_packed_triangular(n_parties, chol_packed, lower=True)
            cov = pm.Deterministic("sigma_rw", pt.dot(chol, chol.T))
            import pdb;pdb.set_trace()

            mu_0 = pm.Normal("mu_0", mu=0, sigma=0.1, shape=(n_parties,)) # pm.Normal("mu_0", mu=0, sigma=0.1, shape=(n_parties,))
            init_dist = pm.MvNormal.dist(mu_0, cov = pt.eye(n_parties) * 0.1)
            n_days = max(time_idx) + 1
            swings_full = pm.MvGaussianRandomWalk("swings_full", mu=pt.zeros(n_parties), cov=cov, shape=(n_days, n_parties), init_dist=init_dist)
            swings = swings_full[time_idx]
            #import pdb;pdb.set_trace()

            # Observation noise (polling error, scaled by sample size)
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=1, shape=n_parties)  # Base polling error
            sigma_scaled = sigma_obs / pt.sqrt(sample_size[:, None])  # Scale by sample size
            #import pdb;pdb.set_trace()

            # Likelihood
            Y_obs = pm.Normal("Y_obs", mu=swings, sigma=sigma_scaled, observed=Y)
            #import pdb;pdb.set_trace()

            # Run MCMC
            trace = pm.sample(500, tune=1000, target_accept=0.85,  max_treedepth=10,cores=4, return_inferencedata=True)
        print(trace)
        print("success")

        # ---- Analyze Results ----
        az.plot_posterior(trace, var_names=["sigma_rw", "sigma_obs"])
        az.plot_trace(trace)
        # Trace contains the inferred parameters of the model.

        import pdb;pdb.set_trace()
