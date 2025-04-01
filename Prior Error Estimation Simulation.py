import pandas as pd
import numpy as np
import os,time
import io
import os
import glob
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal, dirichlet

from scipy.stats import multivariate_t
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize


# automatic error debugging
import sys
import pdb
import traceback

def exception_handler(type, value, tb):
    traceback.print_exception(type, value, tb)  # Print the error as usual
    print("\n--- Entering post-mortem debugging ---\n")
    pdb.pm()  # Start debugger at the error location

sys.excepthook = exception_handler



base_dir = Path('C:\\Dania\\2024\\Australian Election') if os.name == "nt" else Path.home() / "Australian Election"
os.chdir(base_dir)






start = time.time()



def get_prior_ALR_covariance(data_year):
    next_year = str(int(data_year)+3)
    Actual_results = pd.read_csv(f"{next_year}HouseDOPByDivision.csv", skiprows=1, index_col = None).rename(columns={'DivisionNm':'div_nm'})
    # COUntNUmber ==0, Pref Percent & decide on format - long or wide? Will generate swings for each, so wide is best


    # Need the following: dict of new_div: party_First_Pref_votes_in_alphabetical_order (separate INDXs and COALs)
    Actual_results = Actual_results.loc[(Actual_results['CountNumber']==0) & (Actual_results['CalculationType']=='Preference Percent'),['div_nm','PartyAb','CalculationValue']]
    Actual_results.loc[Actual_results['PartyAb'].isna(),].fillna('IND')
    Actual_results.loc[Actual_results['PartyAb']=='GVIC','PartyAb'] = 'GRN'
    Actual_results.loc[Actual_results['PartyAb']=='CLR','PartyAb'] = 'ALP'

    # rename IND to INDX by order

    target = 'IND'

    Actual_results_dict = {}
    Fundamentals_results_dict = {}
    Fundamentals_estimate_dict = {}

    Prior_estimates_df = pd.read_csv(f"Fundamentals_Votes_For_{next_year}.csv", index_col = None)

    Prior_estimates_dict = {
        div: pd.DataFrame([group.set_index("PartyAb")["FP_Votes"].to_dict()])
        for div, group in Prior_estimates_df.groupby("div_nm")
    }

    def group_into_Fundamentals_Categories(party_votes_shares_df, div, is_Other = True):
        # creates a structured data frame  with columns ALP,COAL,GRN,Other by combining all the votes of the respective categories

        ALP_cat = {'ALP','CLR'}
        COAL_cat = {'COAL','COALNP','COALLP','LP','NP','CLP','LNP','LNQ'}
        GRN_cat = {'GRN'}

        Non_Other_sets = ALP_cat | COAL_cat | GRN_cat  # Union of all sets
        Other_cols = set(party_votes_shares_df.columns) - Non_Other_sets  # Columns in none of the sets

        ALPs = ALP_cat.intersection(party_votes_shares_df.columns)
        COALs = COAL_cat.intersection(party_votes_shares_df.columns)
        GRNs = GRN_cat.intersection(party_votes_shares_df.columns)
        OTHs = Other_cols

        # Compute the sums
        sum1 = party_votes_shares_df[list(next(iter(ALPs)) if len(ALPs) == 1 and isinstance(next(iter(ALPs)), set) else ALPs)].sum(axis=1).iloc[0]
        sum2 = party_votes_shares_df[list(next(iter(COALs)) if len(COALs) == 1 and isinstance(next(iter(COALs)), set) else COALs)].sum(axis=1).iloc[0]
        if is_Other:
            sum3 = party_votes_shares_df[list(next(iter(GRNs)) if len(GRNs) == 1 and isinstance(next(iter(GRNs)), set) else GRNs)].sum(axis=1).iloc[0]
            sum4 = party_votes_shares_df[list(next(iter(OTHs)) if len(OTHs) == 1 and isinstance(next(iter(OTHs)), set) else OTHs)].sum(axis=1).iloc[0]
        else:
            sum3 = party_votes_shares_df[list(next(iter(GRNs)) if len(GRNs) == 1 and isinstance(next(iter(GRNs)), set) else GRNs) + list(next(iter(OTHs)) if len(OTHs) == 1 and isinstance(next(iter(OTHs)), set) else OTHs)].sum(axis=1).iloc[0]

        if is_Other:
            Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'GRN':sum3,'Other':sum4}], index=[div])
        else:
            Fundamentals_grouped_df = pd.DataFrame([{'ALP':sum1,'COAL':sum2,'Other':sum3}], index=[div])


        return Fundamentals_grouped_df

    Fundamentals_results_list = []
    Fundamentals_estimate_list = []


    for div in Actual_results['div_nm'].unique():
        div_results = Actual_results.loc[Actual_results['div_nm'] == div,].copy()

        div_results.loc[:,'Count'] = div_results.groupby('PartyAb').cumcount() + 1     # Count instances of the target string
        # Replace duplicates of the target string with increasing strings IND1, IND2, IND3, ...
        adjusted_party_names = div_results.apply(
            lambda row: f"{row['PartyAb']}{row['Count']}" if row['PartyAb'] == target else row['PartyAb'], axis=1
        ).reset_index(drop=True)

        div_results_combined = div_results.groupby(['div_nm', 'PartyAb'], as_index=False)['CalculationValue'].sum()

        Actual_results.loc[Actual_results['div_nm'] == div,'PartyAb'] = adjusted_party_names

        Actual_results_dict[div] = div_results_combined.pivot(index='div_nm', columns='PartyAb', values='CalculationValue')
        #Fundamentals_results_dict[div] = group_into_Fundamentals_Categories(Actual_results_dict[div], div)
        #Fundamentals_estimate_dict[div] = group_into_Fundamentals_Categories(Prior_estimates_dict[div], div)

        Fundamentals_results_list.append(group_into_Fundamentals_Categories(Actual_results_dict[div], div))
        Fundamentals_estimate_list.append(group_into_Fundamentals_Categories(Prior_estimates_dict[div], div))



    Fundamentals_results_df = pd.concat(Fundamentals_results_list)/100
    Fundamentals_estimate_df = pd.concat(Fundamentals_estimate_list)

    import pdb; pdb.set_trace()

    # Gorton 2016 adjustment: add 0.001 from GRN to Other
    Fundamentals_results_df.loc[Fundamentals_results_df['Other'] == 0.0,['GRN','Other']] += (-0.001,0.001)
    Fundamentals_estimate_df.loc[Fundamentals_estimate_df['Other'] == 0.0,['GRN','Other']] += (-0.001,0.001)


    Fundamentals_results_df = Fundamentals_results_df.div(Fundamentals_results_df.sum(axis=1), axis=0)
    Fundamentals_estimate_df = Fundamentals_estimate_df.div(Fundamentals_estimate_df.sum(axis=1), axis=0)  

    Fundamentals_results_df.index = data_year +  Fundamentals_results_df.index
    Fundamentals_estimate_df.index = data_year +  Fundamentals_estimate_df.index


    return Fundamentals_results_df, Fundamentals_estimate_df

Fundamentals_results_list = []
Fundamentals_estimate_list = []


for data_year in ['2013','2016','2019']:
    Fundamentals_results_df,Fundamentals_estimate_df = get_prior_ALR_covariance(data_year)
    Fundamentals_results_list.append(Fundamentals_results_df)
    Fundamentals_estimate_list.append(Fundamentals_estimate_df)

full_Fundamentals_results_df = pd.concat(Fundamentals_results_list)
full_Fundamentals_estimate_df = pd.concat(Fundamentals_estimate_list)

# center the natural swings to avoid bias - we assume swings are generally unbiased - that we cannot predict direction 3 years prior!
swings = full_Fundamentals_results_df - full_Fundamentals_estimate_df
#swings_centered = (swings - swings.mean()).sum(axis=1)
full_Fundamentals_estimate_df = full_Fundamentals_estimate_df + swings.mean()

import pdb; pdb.set_trace()


ref_col = 'COAL'  # Reference category to remove
ref_val = full_Fundamentals_estimate_df.columns.get_loc(ref_col)

results_alr = np.log(full_Fundamentals_results_df.drop(columns=[ref_col]).div(full_Fundamentals_results_df[ref_col], axis=0))
estimate_alr = np.log(full_Fundamentals_estimate_df.drop(columns=[ref_col]).div(full_Fundamentals_estimate_df[ref_col], axis=0))
alr_swing = results_alr - estimate_alr

alr_swing_cov = alr_swing.cov()


print(alr_swing.cov())
print((full_Fundamentals_results_df - full_Fundamentals_estimate_df).mean()) # should be 0 due to centralisation adjustment


simulated_alr_swings = multivariate_normal.rvs(mean=np.zeros(alr_swing.shape[1]), cov=alr_swing_cov, size=451)

predicted_alr = estimate_alr + simulated_alr_swings




import pdb;pdb.set_trace()




def split_vote_share_dirichlet(total_vote_share, Others_proportions, alpha_scale=50, n_samples=1): # chatgpt written
    """
    Splits a given total vote share across m parties using a Dirichlet distribution.
    
    Parameters:
    - total_vote_share (float): The total percentage of votes to split (e.g., 0.185 for 18.5%).
    - proportions (list of float): A list of m proportions summing to 1 (used as mean proportions).
    - alpha_scale (float): Scaling factor for the Dirichlet concentration parameter (higher = lower variance).
    - n_samples (int): Number of samples to generate.
    
    Returns:
    - np.ndarray: An array of shape (n_samples, m) with sampled vote shares.
    """
    assert np.isclose(sum(Others_proportions), 1), "Proportions must sum to 1."
    alpha = np.array(Others_proportions) * alpha_scale  # Convert mean proportions into Dirichlet parameters
    samples = np.random.dirichlet(alpha, size=n_samples) * total_vote_share
    return samples



#def alr_to_simplex(alr_samples): # chatgpt written
#    """Convert ALR-transformed samples back to simplex space."""
#    exp_values = np.exp(alr_samples)
#    denominator = 1 + np.sum(exp_values, axis=-1, keepdims=True)
#    base_category = 1 / denominator
#    return np.hstack([exp_values / denominator, base_category])

def alr_to_simplex_vectorized(df, ref_col):
    """Inverse ALR transformation for an entire DataFrame in a vectorized way."""
    # Convert the DataFrame to a numpy array for vectorized operations
    alr_vals = df.values
    
    # Apply the inverse ALR transformation across all values
    exp_vals = np.exp(alr_vals)

    # Compute the reference category correctly
    ref_vals = 1 / (1 + np.sum(exp_vals, axis=1, keepdims=True))  # Shape: (n_samples, 1)

    # Compute all components
    simplex_vals = np.concatenate((exp_vals * ref_vals, ref_vals), axis=1)  # Shape: (n_samples, D)
    
    # Create new column names, appending a reference category
    new_columns = df.columns.tolist() + [ref_col]
    
    # Return as a DataFrame with the original indices and new columns
    return pd.DataFrame(simplex_vals, columns=new_columns, index=df.index)

def split_category_in_simplex(simplex_probs, split_index, num_splits, dirichlet_alpha): # chatgpt written
    """Splits a category in probability space into `num_splits` subcategories using a Dirichlet distribution."""
    category_to_split = simplex_probs[:, split_index]
    split_proportions = dirichlet.rvs(dirichlet_alpha, size=len(simplex_probs))
    new_split_probs = category_to_split[:, None] * split_proportions

    # Construct new probability simplex
    new_simplex_probs = np.delete(simplex_probs, split_index, axis=1)
    new_simplex_probs = np.hstack([
        new_simplex_probs[:, :split_index], 
        new_split_probs, 
        new_simplex_probs[:, split_index:]
    ]) 
    return new_simplex_probs

prior_prediction = alr_to_simplex_vectorized(predicted_alr, ref_col)[['ALP','COAL','GRN','Other']]

def simulate_swings(alr_swing, alr_swing_cov, prior_prediction, num_simulations, dist, df_t):

    all_simulated_samples = np.zeros((num_simulations, prior_prediction.shape[0], prior_prediction.shape[1]))
    alr_simulated_samples = np.zeros((num_simulations, alr_swing.shape[0], alr_swing.shape[1]))

    for i in range(num_simulations):

        if dist == 't':
            # Rescale covariance for the multivariate t-distribution
            cov_t_corrected = ((df_t - 2) / df_t) * alr_swing_cov  # Adjust scale matrix
            # Simulate swings using the corrected covariance
            simulated_alr_swings = multivariate_t.rvs(loc=np.zeros(alr_swing.shape[1]), shape=cov_t_corrected, df=df_t, size=451)
        elif dist == 'Normal':
            simulated_alr_swings = multivariate_normal.rvs(mean=np.zeros(alr_swing.shape[1]), cov=alr_swing_cov, size=451)

        predicted_alr = estimate_alr + simulated_alr_swings

        curr_prediction = alr_to_simplex_vectorized(predicted_alr, ref_col)[['ALP','COAL','GRN','Other']]
        prior_prediction += curr_prediction

        all_simulated_samples[i] = curr_prediction
        alr_simulated_samples[i] = predicted_alr

    

    prior_prediction /= 1000

    simulated_variance = np.var(all_simulated_samples, axis=0)

    # Compute the standard deviation as the square root of the variance
    simulated_std_dev = np.sqrt(simulated_variance)

    return prior_prediction, pd.DataFrame(simulated_std_dev, columns=full_Fundamentals_estimate_df.columns, index=full_Fundamentals_estimate_df.index), all_simulated_samples, alr_simulated_samples


df_t = 5
num_simulations = 1000

t_means, t_stds, all_simulated_samples_t, alr_simulated_samples_t = simulate_swings(alr_swing, alr_swing_cov, prior_prediction, num_simulations, 't', df_t)

N_means, N_stds, all_simulated_samples_normal, alr_simulated_samples_normal = simulate_swings(alr_swing, alr_swing_cov, prior_prediction, num_simulations, 'Normal', df_t)

import seaborn as sns
import matplotlib.pyplot as plt

sns.kdeplot(alr_simulated_samples_normal[:, 0], label="Normal", color="blue")
sns.kdeplot(alr_simulated_samples_t[:, 0], label="T-distribution", color="red", linestyle="dashed")
plt.legend()
plt.title("Comparison of Simulated Swings (ALR Component 1)")
plt.show()

    






import pdb;pdb.set_trace()

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.special import gammaln  # Correct import for the gamma function

# Load your ALR-transformed swings (shape: 150 electorates × 3 ALR components)

# Step 1: Fit a Multivariate Normal Distribution
mu_norm = np.mean(alr_swing, axis=0)
cov_norm = np.cov(alr_swing, rowvar=False)  # Covariance of ALR swings
log_likelihood_norm = np.sum(stats.multivariate_normal.logpdf(alr_swing, mean=mu_norm, cov=cov_norm))

# Step 2: Compute the log-likelihood for a multivariate t-distribution
def multivariate_t_logpdf(x, df, mu, cov):
    """
    Compute the log-density of a multivariate t-distribution at x.
    """
    d = len(mu)  # Number of dimensions (should be 3)
    x_mu = x - mu  # Center the data
    inv_cov = np.linalg.inv(cov)  # Inverse covariance matrix
    quad_form = np.sum(x_mu @ inv_cov * x_mu, axis=1)  # Quadratic form (Mahalanobis distance)

    log_det_cov = np.linalg.slogdet(cov)[1]  # Log determinant of covariance matrix
    log_const = (
        gammaln((df + d) / 2)  # Corrected gamma function
        - gammaln(df / 2)
        - (d / 2) * np.log(df)
        - (d / 2) * np.log(np.pi)
        - 0.5 * log_det_cov
    )
    
    log_pdf = log_const - ((df + d) / 2) * np.log(1 + quad_form / df)
    return log_pdf

def multivariate_t_log_likelihood(params):
    """ Negative log-likelihood function for a multivariate t-distribution. """
    df, *mu_flat, cov_flat = params[0], params[1:4], params[4:]
    mu = np.array(mu_flat)
    cov = np.reshape(cov_flat, (3, 3))
    
    if df <= 2:  # Degrees of freedom must be >2 for a valid covariance
        return np.inf

    log_likelihood = np.sum(multivariate_t_logpdf(alr_swing, df, mu, cov))
    return -log_likelihood  # Minimize negative log-likelihood

# Step 3: Optimize parameters (df=5, mean=normal mean, covariance=normal covariance)
init_params = np.hstack([5, mu_norm, cov_norm.flatten()])
bounds = [(2.01, None)] + [(None, None)] * 3 + [(None, None)] * 9  # Ensure df > 2

res = minimize(multivariate_t_log_likelihood, init_params, bounds=bounds, method="L-BFGS-B")

df_t, mu_t, cov_t = res.x[0], res.x[1:4], np.reshape(res.x[4:], (3, 3))
log_likelihood_t = -res.fun

# Step 4: Compare Log-Likelihoods
print(f"Log-likelihood (Normal): {log_likelihood_norm}")
print(f"Log-likelihood (T): {log_likelihood_t}")
print(f"Estimated Degrees of Freedom (T): {df_t:.2f}")

if log_likelihood_t > log_likelihood_norm:
    print("Multivariate t-distribution fits better!")
else:
    print("Multivariate normal fits better!")



import pdb;pdb.set_trace()



