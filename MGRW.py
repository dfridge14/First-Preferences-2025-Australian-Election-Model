import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd
import os
import arviz as az
from datetime import datetime
from matplotlib import pyplot as plt

from scipy.linalg import cholesky
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

#pt.config.mode = 'NUMBA'


os.chdir("C:\\Dania\\ABS downloads")

Poll_Swings_National = pd.read_csv("2022PollSwingsNational.csv", index_col = None)

Y = Poll_Swings_National.iloc[:,2:8].values
sample_size = Poll_Swings_National['Sample size'].values
time_idx = Poll_Swings_National['Date'].values

#import pdb;pdb.set_trace()

# Assuming df_polling is your dataframe
# We will use the polling data for the 5 electorates
#polling_data = Poll_Swings_National.values  # Convert DataFrame to NumPy array (2D)
#1 = (df_polling.index - df_polling.index[0]).days  # Convert 1 to 'days since start'



n_parties = 6

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
