import pymc as pm
import pytensor.tensor as pt
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

data_year = '2016'

Poll_Swings_National = pd.read_csv(f"{data_year}PollSwingsNational.csv", index_col = None)

Y = Poll_Swings_National.iloc[:,2:8].values
sample_size = Poll_Swings_National['Sample size'].values
time_idx = Poll_Swings_National['Date'].values

#import pdb;pdb.set_trace()

# Assuming df_polling is your dataframe
# We will use the polling data for the 5 electorates
#polling_data = Poll_Swings_National.values  # Convert DataFrame to NumPy array (2D)
#1 = (df_polling.index - df_polling.index[0]).days  # Convert 1 to 'days since start'



n_parties = 6

# Bayesian Multivariate Gaussian Random Walk for Polling Data
# Using Stan and Python (CmdStanPy for efficiency)

import numpy as np
import pandas as pd
import cmdstanpy
import matplotlib.pyplot as plt
import arviz as az

# Step 1: Load Data (Placeholder, replace with actual data)
# Columns: [day, sample_size, party_1, party_2, party_3, party_4]
df = pd.read_csv('polling_data.csv')

# Step 2: Choose Reference Category (e.g., the largest average party)
mean_shares = df.iloc[:, 2:].mean()
ref_party = mean_shares.idxmax()  # or idxmin() for smallest

# Step 3: ALR Transformation (remove reference party)
def alr_transform(vote_shares, ref_col):
    alr_values = np.log(vote_shares.drop(columns=[ref_col]).values) - np.log(vote_shares[ref_col].values[:, None])
    return alr_values

alr_votes = alr_transform(df.iloc[:, 2:], ref_party)
days = df['day'].values
sample_sizes = df['sample_size'].values

# Step 4: Infer Poll Variance (multinomial-derived covariance)
def poll_covariance(proportions, sample_size):
    k = len(proportions)
    cov_matrix = np.diag(proportions) - np.outer(proportions, proportions)
    return cov_matrix / sample_size

poll_covs = np.array([poll_covariance(row, n) for row, n in zip(df.iloc[:, 2:].values, sample_sizes)])

# Step 5: Stan Model Definition (Multivariate Gaussian RW with poll uncertainty)
stan_code = """
data {
    int<lower=1> T;  // Number of time points
    int<lower=1> K;  // Number of parties (after ALR transformation)
    matrix[T, K] y;  // Observed ALR vote shares
    matrix[K, K] Sigma_poll[T];  // Polling error covariance per day
}
parameters {
    matrix[T, K] x;  // Latent true ALR vote shares
    vector[K] mu;  // Mean drift (long-term trend)
    cov_matrix[K] Sigma_rw;  // RW covariance
}
model {
    // Prior on trend
    mu ~ normal(0, 0.1);
    
    // Prior on covariance
    Sigma_rw ~ lkj_corr(2);
    
    // Random walk prior
    for (t in 2:T) {
        x[t] ~ multi_normal(x[t-1] + mu, Sigma_rw);
    }
    
    // Observations
    for (t in 1:T) {
        y[t] ~ multi_normal(x[t], Sigma_poll[t]);
    }
}
"""

# Step 6: Compile & Fit Stan Model
stan_model = cmdstanpy.CmdStanModel(stan_file='polling_rw.stan', model_code=stan_code)
data_dict = {
    'T': len(days),
    'K': alr_votes.shape[1],
    'y': alr_votes,
    'Sigma_poll': poll_covs
}
fit = stan_model.sample(data=data_dict, chains=4, parallel_chains=4, iter_warmup=500, iter_sampling=1000)

# Step 7: Extract & Plot Results
idata = az.from_cmdstanpy(posterior=fit)
plt.figure(figsize=(10, 5))
for i in range(alr_votes.shape[1]):
    az.plot_hdi(days, idata.posterior['x'][:, :, i], color='blue', fill_kwargs={'alpha': 0.3})
    plt.plot(days, np.median(idata.posterior['x'], axis=(0, 1))[:, i], label=f'Party {i+1}')
plt.xlabel('Days Since Election')
plt.ylabel('ALR Vote Share')
plt.legend()
plt.show()


    

def old_model_PyMC():
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
